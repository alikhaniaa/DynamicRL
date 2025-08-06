bcirl-framework/
├── .gitignore
├── README.md
├── evaluate.py
├── requirements.txt
├── train.py
├── configs/
│   ├── ppo_cartpole.yaml
│   └── ppo_mujoco_bci.yaml
└── src/
    ├── agents/
    │   ├── __init__.py
    │   ├── base_agent.py
    │   └── ppo_agent.py
    ├── core/
    │   ├── __init__.py
    │   └── engine.py
    ├── envs/
    │   ├── __init__.py
    │   └── wrappers.py
    ├── hpo/
    │   ├── __init__.py
    │   ├── base_hpo.py
    │   └── bci_handler.py
    └── utils/
        ├── __init__.py
        ├── logging.py
        ├── replay_buffer.py
        └── video.py
