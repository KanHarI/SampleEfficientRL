from setuptools import find_packages, setup

setup(
    name="sample_efficient_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "chess",
        "numpy",
        "torch",
    ],
    extras_require={
        "dev": [
            "black",
            "mypy",
            "pytest",
            "flake8",
            "isort",
        ],
    },
    python_requires=">=3.7",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Sample Efficient Reinforcement Learning Project",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        'console_scripts': [
            'play_simple=SampleEfficientRL.Envs.Deckbuilder.PlayInCli:main',
            'random_walk=SampleEfficientRL.Envs.Deckbuilder.RandomWalkAgent:main',
        ],
    },
)
