# Python Worker 镜像模板

此模板为当前 osc-agent Python 仓库提供断网验证环境，不是通用项目镜像。其他仓库必须复制模板并
把自身运行、构建和测试依赖固定在镜像中；容器运行时不会联网安装依赖。

在仓库根目录执行：

```bash
docker build -t osc-agent-worker:python311 deploy/docker/python311
docker run --rm --network none --read-only --user 65532:65532 \
  --tmpfs /tmp --tmpfs /home/osc \
  osc-agent-worker:python311 \
  /bin/bash --noprofile --norc -c \
  'python --version; pytest --version; python -m pip check'
docker image inspect --format '{{.Id}}' osc-agent-worker:python311
```

把最后一条命令返回的完整小写 `sha256:<64 hex>` 写入 `config.yml` 的 `repositories` 区段。配置禁止使用
tag；Worker 每次领取 Job 都会验证该 ID 在本机存在并精确匹配。

基础镜像 tag 只影响构建过程。正式部署应在管理员验证后记录基础镜像 digest，并在组织维护
的派生 Dockerfile 中固定；最终运行边界始终以本机构建结果的不可变 image ID 为准。

不要把 API Key、GitHub token、私有 pip 配置、Git credential 或任何宿主配置复制进镜像。
如果贡献修改了依赖锁而现有镜像没有所需依赖，验证应当失败，由管理员重建镜像并重新执行
`/osc-agent plan`，不能在断网容器中临时下载。
