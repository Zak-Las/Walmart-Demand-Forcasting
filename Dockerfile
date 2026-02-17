# 1. Base Image
# Use micromamba for faster, lower-memory solves in Dev Containers.
FROM mambaorg/micromamba:1.5.8

# 2. Metadata: Add labels to describe the image
LABEL maintainer="Zak-Las"
LABEL description="A containerized environment for Zak's data science portfolio project."

# 3. Environment Variables: Centralize configuration
# Using ARG and ENV makes the Dockerfile more readable and easier to maintain.
ARG CONDA_ENV_NAME=wdf
ENV CONDA_ENV_NAME=${CONDA_ENV_NAME}
ENV APP_HOME=/workspace
WORKDIR $APP_HOME

# 4. Caching Dependencies: Optimize build time
# Copy only the environment file first and install dependencies.
# This leverages Docker's layer caching. The dependencies will only be re-installed
# if the environment.yml file changes, not every time a project file is modified.
USER root

# Minimal OS deps for common Python wheels on Debian/Ubuntu
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		make \
		build-essential \
		git \
		libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace \
	&& chown -R mambauser:mambauser /workspace /opt/conda

USER mambauser

COPY --chown=mambauser:mambauser environment.devcontainer.yml /tmp/environment.devcontainer.yml
RUN micromamba create -y -n ${CONDA_ENV_NAME} -f /tmp/environment.devcontainer.yml \
	&& micromamba clean -a -y

# 5. Copy Project Files: Add project files after dependency installation
COPY --chown=mambauser:mambauser . .

# 6. Activate Conda Environment: Make the environment's tools available on the PATH
# This makes it easy to run commands like 'python' or 'jupyter' directly.
ENV PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:$PATH
ENV CONDA_DEFAULT_ENV=${CONDA_ENV_NAME}

# Install pip-only runtime deps (pinned for reproducibility)
RUN python -m pip install --no-cache-dir \
	lightgbm==4.6.0 \
	optuna==4.6.0 \
	optuna-integration[lightgbm]==4.6.0 \
	pytorch-lightning==2.5.5 \
	torch==2.8.0 \
	neuralforecast==3.1.2 \
	rich==14.2.0 \
	tqdm==4.67.1 \
	tomli==2.4.0 \
	kaggle==1.7.4.5

# 7. Expose Port: Document the port used by Jupyter
EXPOSE 8888

# 8. Default Command: Start Jupyter Lab for a better user experience
# Jupyter Lab provides a more modern and feature-rich interface than the classic notebook.
# The token is intentionally left blank for ease of access in a trusted environment.
# For a public-facing server, you would want to configure a token or password.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
