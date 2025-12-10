| **Phase**               | **Type**             | **Parameter/Setting**              | **Details**                                                                                     |
|--------------------------|----------------------|-------------------------------------|-------------------------------------------------------------------------------------------------|
| **Pretraining**          | General Settings     | Optimizer                          | AdamW, learning rate 5e-4 (cosine decay, warm-up 1k steps), weight decay 5e-6.                 |
|                          |                      | Drop Path                          | Drop probability: 0.2                                                                          |
|                          |                      | Batch Size                         | 32 GPUs, batch size 1 per GPU.                                                                 |
|                          |                      | Loss Weighting                     | ERA5 = 2.0, GFS-T0 = 1.5, others = 1.                                                          |
|                          | Variable Weights     | Surface                            | MSL = 1.5, 10U = 0.77, 10V = 0.66, 2T = 3.0.                                                   |
|                          |                      | Atmospheric                        | Z,c = 2.8, Q,c = 0.78, T,c = 1.7, U,c = 0.87, V,c = 0.6.                                       |
| **Fine-tuning**          | Variable Weights     | Surface                            | MSL = 1.6, 10U = 0.77, 10V = 0.66, 2T = 3.5.                                                   |
|                          |                      | Atmospheric                        | Z,c = 3.5, Q,c = 0.8, T,c = 1.7, U,c = 0.87, V,c = 0.6.                                        |
|                          | Air Pollution        | Weight Scaling                     | Based on scale / persistence MAE.                                                              |
|                          | CAMS 0.4°            | Learning Rates                     | High: 1e-3 (new variables), Low: 1e-4.                                                         |
|                          |                      | Batch Size                         | 16 GPUs, batch size 1 per GPU.                                                                 |
| **Short Lead-Time Finetuning** | 0.25° Resolution     | Training Steps                     | 8k steps, 2 rollouts.                                                                          |
|                          |                      | Learning Rate                      | Warm-up 1k steps, constant 5e-5.                                                               |
|                          | 0.1° Resolution      | Training Steps                     | 12.5k steps, single-step rollout.                                                              |
|                          |                      | Learning Rate                      | Warm-up 1k steps, constant 2e-4.                                                               |
|                          |                      | Weight Decay                       | Set to 0.                                                                                      |
|                          |                      | Memory Optimization                | Activation checkpointing and gradient sharding.                                                |
| **Rollout Finetuning**   | Replay Buffer        | Buffer Size                        | 0.25°: 4000 samples (20 GPUs); 0.1°: 640 samples (32 GPUs).                                    |
|                          |                      | Buffer Update Period               | Every 10 steps.                                                                                |
|                          | Training Steps       | Steps                              | 0.25°: 13k; 0.1°: 6.25k; CAMS: 6.5k.                                                          |
|                          | Learning Rate        | Fixed                              | 5e-5.                                                                                          |
|                          | Memory Optimization  | LoRA for attention layers, gradient checkpointing, replay buffer.                             |
