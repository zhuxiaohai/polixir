# License 申请教程

本 Starting Kit 中提供了一份直到比赛结束为止有效的公共 License。该教程将指引如何申请一份个人使用的 License。

## 获取机器信息

Revive SDK 中，个人使用的 License 是按机器授权的。下载了 Revive SDK 后，进入 `licenses` 文件夹，获取本机机器信息：

```
cd baseline/revive/licenses
python get_machine_info.py
```

获取的机器信息保存在了`machine_info.json`中。

## 在 Revive 官网获取 License

访问 [Revive 官网](https://revive.cn)，注册/登陆自己的账号，点击网页右上角 `SDK` -> `SDK License申请`，在`上传设备文件`中将`machine_info.json`上传，即可获取到 License 并下载，解压后得到`license.lic`。


## 在计算设备上使用 License

将解压后的得到的`license.lic`文件上传到计算设备的指定位置，并为其配置好环境变量`$PYARMOR_LICENSE`。例如，本 Starting Kit 中数据集中保存在`starting_kit/baseline/data`中，则环境变量应设置为：

```shell
export PYARMOR_LICENSE=$BASELINE_ROOT/data/license.lic
```

其中`$BASELINE_ROOT`是记录了 Baseline 根目录的环境变量。该设置可写在相关 Profile 文件如 `.bashrc` 中，以方便自动配置环境变量。

## FAQ

* 获取到的`machine_info.json`内容基本是空的？

  请确认是否使用的是 MAC OS 系统。若符合，则由于 MAC OS 系统缺少的 `nohup` 命令，机器信息没有被正确读取，这将在后续的 Revive SDK 中修复。

* 重启电脑后，运行 Revive 报 `License is not for this machine` 的错误？

  请确认是否使用的是 WSL。若符合，则由于 License 主要是根据 MAC 地址来识别设备的，而 WSL 在每次 Windows 重启后都会更换 IP 与 MAC 地址，因此造成 License 不识别为同一台设备。

* 如何获取上传的 `machine_info.json`？

  `machine_info.json` 上传后，保存在 Revive 服务器的用户存储区域中。可以通过以下步骤获取上传的JSON文件：

  1. 访问 Revive 官网 -> `控制台` -> `存储管理`；
  2. 访问 `/business/deviceXXXX(32位uuid)/`
  3. 上传的 `machine_info.json` 位于此处。

* 如何重新上传新的 `machine_info.json`？

  Revive 目前仅支持一个账号绑定一个设备信息，上传后无法再次上传。可以通过注册新账号的方式绑定新的 `machine_info.json` 。
