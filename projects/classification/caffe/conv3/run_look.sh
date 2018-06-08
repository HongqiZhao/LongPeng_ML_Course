#########################################################################
# File Name: run_look.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: 2017年06月21日 星期三 22时09分52秒
#########################################################################
#!/bin/bash
./geneaccrefine
./genelossrefine
python show_loss.py
python show_acc.py
