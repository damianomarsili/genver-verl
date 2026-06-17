# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import os
import socket

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env, with_default_ray_init_kwargs
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import load_extern_object


def _task_runner_runtime_env(config) -> dict:
    """Build optional runtime_env overrides for the TaskRunner actor.

    GENVER SAM3 reward executes inside TaskRunner. If the actor has no GPU reservation,
    Ray sets CUDA_VISIBLE_DEVICES to empty by default, which makes torch.cuda unavailable.
    """
    reward_cfg = config.get("custom_reward_function") or {}
    sam3_device = str(reward_cfg.get("genver_sam3_device", "") or "").strip().lower()
    if not sam3_device.startswith("cuda"):
        return {}
    return {"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}}


def _get_hf_trust_remote_code(config) -> bool:
    """Resolve trust_remote_code from data and model config.

    Some multimodal models such as Molmo2 ship their processor through
    remote code under the model path rather than the data config. When
    only the model-side flag is enabled, using data.trust_remote_code
    alone silently drops the processor and later breaks dataset loading.
    """
    data_trust_remote_code = bool(config.data.get("trust_remote_code", False))
    model_trust_remote_code = bool(config.actor_rollout_ref.model.get("trust_remote_code", False))
    return data_trust_remote_code or model_trust_remote_code


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray_init_kwargs = with_default_ray_init_kwargs(
            OmegaConf.to_container(ray_init_kwargs, resolve=True)
        )
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**ray_init_kwargs)

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(
            TaskRunner
        )  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    task_runner_runtime_env = _task_runner_runtime_env(config)

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert (
            is_nvtx_available()
        ), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runtime_env = {"nsight": nsight_options}
        if task_runner_runtime_env:
            runtime_env = OmegaConf.merge(
                OmegaConf.create(runtime_env), OmegaConf.create(task_runner_runtime_env)
            )
            runtime_env = OmegaConf.to_container(runtime_env, resolve=True)
        runner = task_runner_class.options(runtime_env=runtime_env).remote()
    else:
        if task_runner_runtime_env:
            runner = task_runner_class.options(runtime_env=task_runner_runtime_env).remote()
        else:
            runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # use new model engine implementation
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = (
                lora_rank > 0
                or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            )
            # NOTE: In new model engine, ref policy and actor rollout are in same ActorRolloutRefWorker,
            # while in legacy model engine, ref policy is in a separate ActorRolloutRefWorker.
            if need_reference_policy(config) and not ref_in_actor:
                role = Role.ActorRolloutRef
            else:
                role = Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        # Note: sync mode validation is now handled in RolloutConfig.__post_init__
        # Always use async worker since sync mode is deprecated and rejected
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                # we don't need to specialize critic worker. Just use TrainingWorker
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
                print("Using new worker implementation")
            else:
                raise ValueError(
                    f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}"
                )

        elif config.critic.strategy == "megatron":
            # TODO: switch this to TrainingWorker as well
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError(
                    "config.reward_model.n_gpus_per_node must be greater than 0"
                )
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [
                config.reward_model.n_gpus_per_node
            ] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=self.mapping
        )
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get(
                "use_legacy_worker_impl", "auto"
            )
            if use_legacy_worker_impl in ["auto", "enable", "disable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            # elif use_legacy_worker_impl == "disable":
            #     from verl.workers.engine_workers import RewardModelWorker
            #
            #     print("Using new worker implementation")
            else:
                raise ValueError(
                    f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}"
                )

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role

        # Ref policy has been fused into ActorRolloutRefWorker in new model engine,
        # we don't need to add a separate ref policy worker group.
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            return

        if need_reference_policy(config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = _get_hf_trust_remote_code(config)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )

        val_reward_config = config
        val_custom_reward_cfg = config.get("val_custom_reward_function", None)
        if val_custom_reward_cfg is not None:
            val_reward_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            val_reward_config.custom_reward_function = OmegaConf.create(
                OmegaConf.to_container(val_custom_reward_cfg, resolve=True)
            )
            print("Using val_custom_reward_function override for validation reward manager.")

        val_reward_fn = load_reward_manager(
            val_reward_config,
            tokenizer,
            num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()


def create_rl_dataset(
    data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1
):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """

    from verl.utils.dataset.rl_dataset import get_dataset_class

    # Get the dataset class
    dataset_cls = get_dataset_class(data_config)

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import math
    from collections import deque

    import torch
    from torch.utils.data import SequentialSampler, Sampler

    # torch.utils.data.RandomSampler could not recover properly
    from torchdata.stateful_dataloader.sampler import RandomSampler

    class _BalancedBinarySampler(Sampler[int]):
        """Sampler that enforces 50/50 sampling between two label groups per batch."""

        def __init__(
            self,
            dataset,
            batch_size: int,
            positive_labels,
            negative_labels,
            seed: int,
        ):
            if batch_size % 2 != 0:
                raise ValueError(
                    f"Balanced binary sampling requires an even batch size, got {batch_size}."
                )
            self.dataset = dataset
            self.batch_size = batch_size
            self.samples_per_class = batch_size // 2
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

            self.positive_labels = self._normalize_label_set(
                positive_labels, "positive"
            )
            self.negative_labels = self._normalize_label_set(
                negative_labels, "negative"
            )

            self.positive_indices = self._collect_indices(self.positive_labels)
            self.negative_indices = self._collect_indices(self.negative_labels)

            if len(self.positive_indices) == 0 or len(self.negative_indices) == 0:
                raise ValueError(
                    "Balanced binary sampling requires at least one example for each label group."
                )

            self.num_batches = math.ceil(
                max(len(self.positive_indices), len(self.negative_indices))
                / self.samples_per_class
            )
            self.length = self.num_batches * self.batch_size

            self._positive_queue: deque[int] = deque()
            self._negative_queue: deque[int] = deque()

            print(
                f"Using balanced binary sampler with {len(self.positive_indices)} positive "
                f"and {len(self.negative_indices)} negative examples."
            )

        @staticmethod
        def _normalize_label_set(labels, label_kind: str) -> set[str]:
            if labels is None:
                return set()
            if isinstance(labels, (str, int, float, bool)):
                labels_iterable = [labels]
            else:
                try:
                    labels_iterable = list(labels)
                except TypeError:
                    raise TypeError(
                        f"balance_binary_{label_kind}_labels must be a string or iterable, got {type(labels)}"
                    ) from None

            normalized: set[str] = set()
            for label in labels_iterable:
                if label is None:
                    continue
                normalized.add(str(label).strip().lower())
            if not normalized:
                raise ValueError(
                    f"balance_binary_{label_kind}_labels must contain at least one non-empty label."
                )
            return normalized

        def _collect_indices(self, label_set: set[str]) -> list[int]:
            indices: list[int] = []
            dataframe = getattr(self.dataset, "dataframe", None)
            if dataframe is None:
                for idx in range(len(self.dataset)):
                    example = self.dataset[idx]
                    label = self._extract_label(example)
                    if label in label_set:
                        indices.append(idx)
                return indices

            for idx in range(len(dataframe)):
                entry = dataframe[idx]
                label = self._extract_label(entry)
                if label in label_set:
                    indices.append(idx)
            return indices

        @staticmethod
        def _extract_label(example) -> str | None:
            reward_info = None
            if isinstance(example, dict):
                reward_info = example.get("reward_model")
            if isinstance(reward_info, dict):
                ground_truth = reward_info.get("ground_truth")
                if isinstance(ground_truth, str):
                    return ground_truth.strip().lower()
                if isinstance(ground_truth, (int, float, bool)):
                    return str(ground_truth).strip().lower()
            return None

        def __len__(self) -> int:
            return self.length

        def __iter__(self):
            for _ in range(self.num_batches):
                positive_batch = self._draw_samples(
                    self._positive_queue, self.positive_indices
                )
                negative_batch = self._draw_samples(
                    self._negative_queue, self.negative_indices
                )
                combined = positive_batch + negative_batch
                perm = torch.randperm(len(combined), generator=self.generator).tolist()
                for idx in perm:
                    yield combined[idx]

        def _draw_samples(
            self, queue: deque[int], source_indices: list[int]
        ) -> list[int]:
            while len(queue) < self.samples_per_class:
                queue.extend(self._shuffled_indices(source_indices))
            return [queue.popleft() for _ in range(self.samples_per_class)]

        def _shuffled_indices(self, indices: list[int]) -> list[int]:
            order = torch.randperm(len(indices), generator=self.generator).tolist()
            return [indices[i] for i in order]

    if (
        data_config.sampler is not None
        and data_config.sampler.get("class_path", None) is not None
    ):
        curriculum_class = load_extern_object(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )
    elif data_config.get("balance_binary_labels", False):
        batch_size = data_config.get("gen_batch_size", data_config.train_batch_size)
        positive_labels = list(
            data_config.get("balance_binary_positive_labels", ["yes"])
        )
        negative_labels = list(
            data_config.get("balance_binary_negative_labels", ["no"])
        )
        sampler = _BalancedBinarySampler(
            dataset=dataset,
            batch_size=batch_size,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            seed=data_config.get("seed", 1),
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(
            data_source=dataset, generator=train_dataloader_generator
        )
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
