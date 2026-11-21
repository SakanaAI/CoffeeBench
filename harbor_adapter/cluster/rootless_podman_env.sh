#!/bin/sh
# Compute-node setup for rootless podman (source this file). Requires a podman-static install
# (https://github.com/mgoltzsche/podman-static) unpacked at $PODMAN_STATIC (default $HOME/podman-static4).
set -u
PODMAN_STATIC="${PODMAN_STATIC:-$HOME/podman-static4}"
export PATH=$PODMAN_STATIC/usr/local/bin:$HOME/bin:$HOME/.local/bin:$PATH

# exec可能なnode-local領域を探索 (/dev/shmはnoexecのため不可)
exec_ok() {
  mkdir -p "$1" 2>/dev/null && cp /bin/true "$1/__exectest" 2>/dev/null && "$1/__exectest" 2>/dev/null
}
PODMAN_BASE=""
for cand in "${PBS_LOCALDIR:-/nonexistent}" "/local/$USER" "/local1/$USER" "/local2/$USER" "/tmp/$USER"; do
  d="$cand/cb_${PBS_JOBID:-manual}"
  if exec_ok "$d"; then PODMAN_BASE="$d"; break; fi
done
if [ -z "$PODMAN_BASE" ]; then echo "FATAL: no executable local storage found" >&2; grep -E "shm|/tmp|/local" /proc/mounts >&2; exit 9; fi
echo "PODMAN_BASE=$PODMAN_BASE"
grep -E "shm| /tmp | /local" /proc/mounts || true
STORAGE_DRIVER=vfs
mkdir -p "$PODMAN_BASE"

export XDG_RUNTIME_DIR=$PODMAN_BASE/xdg
mkdir -p "$XDG_RUNTIME_DIR"
export TMPDIR=$PODMAN_BASE/tmp
mkdir -p "$TMPDIR"

export CONTAINERS_CONF=$PODMAN_BASE/containers.conf
cat > "$CONTAINERS_CONF" <<CCEOF
[engine]
helper_binaries_dir = ["$PODMAN_STATIC/usr/local/lib/podman", "$PODMAN_STATIC/usr/local/bin"]
cgroup_manager = "cgroupfs"
events_logger = "file"
events_logfile_path = "$PODMAN_BASE/events.log"
tmp_dir = "$PODMAN_BASE/libpod_tmp"
static_dir = "$PODMAN_BASE/libpod_static"
volume_path = "$PODMAN_BASE/volumes"
[network]
network_backend = "cni"
cni_plugin_dirs = ["$HOME/podman-static4/usr/local/lib/cni"]
network_config_dir = "$PODMAN_BASE/cni"
CCEOF
mkdir -p "$PODMAN_BASE/cni"
cp $HOME/podman-static4/etc/cni/net.d/*.conflist "$PODMAN_BASE/cni/" 2>/dev/null || true
mkdir -p "$PODMAN_BASE/libpod_tmp" "$PODMAN_BASE/libpod_static" "$PODMAN_BASE/volumes"

export CONTAINERS_STORAGE_CONF=$PODMAN_BASE/storage.conf
cat > "$CONTAINERS_STORAGE_CONF" <<EOF
[storage]
driver = "$STORAGE_DRIVER"
graphroot = "$PODMAN_BASE/root"
rootless_storage_path = "$PODMAN_BASE/root"
runroot = "$PODMAN_BASE/run"
[storage.options.overlay]
mount_program = "$PODMAN_STATIC/usr/local/bin/fuse-overlayfs"
ignore_chown_errors = "true"
[storage.options.vfs]
ignore_chown_errors = "true"
EOF

podman --storage-driver "$STORAGE_DRIVER" --storage-opt "$STORAGE_DRIVER.ignore_chown_errors=true" --root "$PODMAN_BASE/root" --runroot "$PODMAN_BASE/run" system service --time=0 unix://$PODMAN_BASE/podman.sock &
PODMAN_SERVICE_PID=$!
sleep 5
export DOCKER_HOST=unix://$PODMAN_BASE/podman.sock
# buildx/buildkitを使わせない (rootless cgroup v1でbuildkitコンテナが起動不能)
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
rm -rf $HOME/.docker/buildx 2>/dev/null || true
export PODMAN_BASE PODMAN_SERVICE_PID
