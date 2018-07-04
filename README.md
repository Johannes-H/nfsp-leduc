# NFSP in Leduc Holdem

This project implements [Neural Fictitious Self-Play](https://arxiv.org/abs/1603.01121) in Leduc Holdem.

## Install
This implementation requires [Torch/LuaJIT](http://torch.ch/), as well as
the following dependencies:
```bash
luarocks install class
```
Also, the 'fsp' module should be in the Lua path, e.g.
```bash
ln -s "$(pwd)"/fsp ~/torch/install/share/lua/5.1/
```

## Run
This is intended to run on cpu, e.g. an AWS c5.large instance reaches 0.1 exploitability in about 5 hours.
```bash
th leduc_fsp3.lua <log_path>
```
