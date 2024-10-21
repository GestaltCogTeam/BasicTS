import torch
import numpy as np
import argparse
import time
import util
from pipeline import train_pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='/data/pems_data/pems_vldb/long_term',help='data path')
parser.add_argument('--input_length',type=int,default=2016,help='')
parser.add_argument('--output_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=3,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=9638,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=1,help='batch size')
parser.add_argument('--tiny_batch_size',type=int,default=256,help='tiny batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=1,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='checkpoint/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()

def main():
    # set seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, 
                                   args.input_length, args.output_length)
    scaler = dataloader['scaler']
    tiny_batch_size = args.tiny_batch_size

    print(args)

    trainer = train_pipeline(scaler, args.input_length, args.output_length, args.in_dim, args.num_nodes, 
                             args.nhid, args.dropout, args.learning_rate, args.weight_decay, device)

    print("start training...",flush=True)
    his_loss =[]
    train_time = []
    val_time = []
    
    for i in range(1, args.epochs+1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            B, T, N, F = x.shape
            batch_num = int(B * N / tiny_batch_size)
            idx_perm = np.random.permutation([i for i in range(B*N)])
            for j in range(batch_num):
                if j==batch_num-1:
                    x_ = x[:, :, idx_perm[(j+1)*tiny_batch_size:], :]
                    y_ = y[:, :, idx_perm[(j+1)*tiny_batch_size:], :]
                else:
                    x_ = x[:, :, idx_perm[j*tiny_batch_size:(j+1)*tiny_batch_size], :]
                    y_ = y[:, :, idx_perm[j*tiny_batch_size:(j+1)*tiny_batch_size], :]

                trainx = torch.Tensor(x_).to(device) # (B, T, N, F)
                trainx = trainx.transpose(1, 2) # (B, N, T, F)
                trainy = torch.Tensor(y_).to(device) # (B, T, N, F)
                trainy = trainy.transpose(1, 2) # (B, N, T, F)
                metrics = trainer.train(trainx, trainy[:,:,:,0])
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                t2 = time.time()
                train_time.append(t2-t1)

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
                # Save the model parameters for subsequent preprocessing
                torch.save(trainer.model.state_dict(), args.save+"linear_transformer.pth")

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            B, T, N, F = x.shape
            batch_num = int(B*N/tiny_batch_size)
            for k in range(batch_num):
                if k==batch_num-1:
                    x_ = x[:, :, (k+1)*tiny_batch_size:, :]
                    y_ = y[:, :, (k+1)*tiny_batch_size:, :]
                else:
                    x_ = x[:, :, k*tiny_batch_size:(k+1)*tiny_batch_size, :]
                    y_ = y[:, :, k*tiny_batch_size:(k+1)*tiny_batch_size, :]
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 2)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 2)
            metrics = trainer.eval(testx, testy[:,:,:,0])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        log = 'Epoch: {:03d}, Validation Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        log = 'Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
        print(log.format(mvalid_loss, mvalid_mape, mvalid_rmse), flush=True)
        val_time.append(s2-s1)
           
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
