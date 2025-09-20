from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='NeuroLens is an AI-powered brain tumor segmentation system designed to overcome the key limitations of existing models. By combining lightweight CNN–Transformer architecture with advanced explainability and federated learning, it delivers fast, accurate, and trustworthy segmentation across diverse MRI scans. Built for real-world clinical use, NeuroLens ensures privacy, generalizability, and interpretability — turning complex brain images into clear, actionable insights.',
    author='Kunal Mallick',
    license='MIT',
)
