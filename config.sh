#!/bin/bash
if [ ! -d "./cache" ]; then
  mkdir ./cache
fi

if [ ! -d "./models" ]; then
  mkdir ./models
fi

# TODO: add folders for: logs, tensorboards

if [ ! -d "./data" ]; then
  mkdir ./data
  # TODO: 下载mat数据
fi