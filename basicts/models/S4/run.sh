dataset="Weather Electricity ETTh1 ETTh2 ETTm1 ETTm2 PEMS03 PEMS04 PEMS07 PEMS08 SD"
ltsf="96" 
for data in $dataset; do
	python experiments/train.py -c baselines/DLinear/$data.py -g 0 
done