#!/bin/bash
script_dir=$(cd "$(dirname "$0")" && pwd)

local_dir=xxx
python  $script_dir/model_merger.py --local_dir $local_dir 
