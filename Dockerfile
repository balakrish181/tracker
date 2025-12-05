# Use Miniconda (has Python available so we can sanitize the YAML)
FROM continuumio/miniconda3:latest

# keep shell as bash for conda activation
SHELL ["bash", "-lc"]

WORKDIR /workspace

# Add your environment.yml
COPY environment_full.yml /tmp/environment.yml

# Sanitize the environment.yml:
#  - remove Windows-only packages and the prefix line
#  - keep pip: block as-is
RUN python - <<'PY'
import yaml
p = "/tmp/environment.yml"
with open(p) as f:
    data = yaml.safe_load(f)
# remove windows-only conda packages if present
win_names = {"vc","ucrt","vs2015_runtime","pywin32","ucrt64"}
deps = data.get("dependencies", [])
new_deps = []
for d in deps:
    if isinstance(d, str):
        name = d.split('=')[0].lower()
        if name in win_names:
            continue
        new_deps.append(d)
    else:
        # keep dict entries (e.g. pip:)
        new_deps.append(d)
data["dependencies"] = new_deps
# drop prefix (Windows local path)
data.pop("prefix", None)
# write sanitized file
with open("/tmp/environment_sane.yml", "w") as f:
    yaml.safe_dump(data, f)
print("Wrote /tmp/environment_sane.yml")
PY

# Create conda environment from the sanitized file
# Use --quiet to reduce spam; --force will recreate if name exists
RUN conda env create -f /tmp/environment_sane.yml -n mob --quiet

# Ensure the environment is activated in subsequent commands
# and available in interactive shells
RUN echo "conda activate mob" >> /etc/profile.d/conda_mob.sh
SHELL ["bash", "-lc"]
ENV PATH /opt/conda/envs/mob/bin:$PATH
ENV CONDA_DEFAULT_ENV mob

# Set working dir and default command
WORKDIR /workspace
CMD [ "bash" ]
