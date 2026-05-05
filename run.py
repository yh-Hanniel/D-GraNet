import argparse
import math
import os
import sys
import time
import torch
import random
import numpy as np
from datetime import datetime
from experiments.exp_forecast import Exp_Forecast
from utils.tools import send_email, find_most_recently_modified_subfolder


# =========================================================
# 辅助函数：生成实验设置 ID (Setting ID)
# [关键修复] 将 des 参数移动到字符串前面，防止因路径过长被截断导致覆盖
# =========================================================
def get_setting(args_, iter_=0):
    if args_.task_name == 'forecasting':
        # 修改：将 args_.des 移动到 task_name 和 model_id 之后
        setting_ = '{}_{}_{}_des-{}_{}_sl-{}_pl-{}_var-{}_dm-{}_stages-{}_{}P{}i_{}P{}i_dec-{}_{}'.format(
            args_.task_name,
            args_.model_id,
            args_.model,
            args_.des,  # <--- 移到这里 (原来在最后)
            args_.data,
            args_.seq_len,
            args_.pred_len,
            args_.enc_in,
            args_.d_model,
            args_.git_multi_stage,
            args_.Patch_layer_num,
            args_.e_layers,
            args_.Patch_layer_num2,
            args_.second_e_layers,
            args_.decoder_cat_num,
            iter_)
    else:
        # 修改：同上
        setting_ = '{}_{}_{}_des-{}_{}_sl-{}_mr-{}_var-{}_dm-{}_stages-{}_{}P{}i_{}P{}i_dec-{}_{}'.format(
            args_.task_name,
            args_.model_id,
            args_.model,
            args_.des,  # <--- 移到这里
            args_.data,
            args_.seq_len,
            args_.mask_rate,
            args_.enc_in,
            args_.d_model,
            args_.git_multi_stage,
            args_.Patch_layer_num,
            args_.e_layers,
            args_.Patch_layer_num2,
            args_.second_e_layers,
            args_.decoder_cat_num,
            iter_)

    # 强制截断到255字符以符合操作系统文件名限制
    return setting_[:255]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='D2GraNet')

    # =========================
    # 1. 基础配置 (Basic Config)
    # =========================
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='D2GraNet', help='model name')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--fix_seed', type=int, default=1, help='fix_seed')
    parser.add_argument('--task_name', type=str, default='forecasting', help='task_name')

    # =========================
    # 2. 数据加载 (Data Loader)
    # =========================
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, etc]')

    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--seasonal_patterns', type=str, default='Yearly', help='subset for M4')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # =========================
    # 3. D2GraNet 模型参数
    # =========================
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size (channels)')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size (channels)')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed_size', type=int, default=8, help='embed_size for D2GraNet embedding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    parser.add_argument('--q_mat_file', type=str, default=None, help='q_mat_file npy file path')
    parser.add_argument('--q_out_mat_file', type=str, default=None, help='q_out_mat_file npy file path')
    parser.add_argument('--Q_MAT_file', type=str, default=None, help='Q_MAT_file npy file')
    parser.add_argument('--Q_OUT_MAT_file', type=str, default=None, help='Q_OUT_MAT_file npy file')
    parser.add_argument('--Q_chan_indep', type=int, default=0, help='Whether to use Channel Independent Q matrix')
    parser.add_argument('--CKA_flag', type=int, default=0, help='CKA analysis flag')

    # =========================
    # 4. 训练控制
    # =========================
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate strategy')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start for OneCycleLR')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--grad_clip', type=int, default=0, help='gradient clipping')
    parser.add_argument('--max_norm', type=float, default=1.0, help='max norm for grad clip')
    parser.add_argument('--save_every_epoch', type=int, default=0, help='save model every epoch')

    parser.add_argument('--train_ratio', type=float, default=1.0, help='percentage of training set used')
    parser.add_argument('--train_part_first', type=int, default=1, help='Use first part of data for training')
    parser.add_argument('--efficient_training', type=int, default=0, help='randomly select channels for training')
    parser.add_argument('--use_norm', type=int, default=1, help='use RevIN norm')

    parser.add_argument('--resume_training', type=int, default=0, help='resume training')
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume epoch')
    parser.add_argument('--checkpoint_check', type=int, default=0, help='if checkpoint exists, skip training')

    # =========================
    # 5. 损失函数
    # =========================
    parser.add_argument('--loss_mode', type=str, default='L1', choices=['L1', 'L2', 'L1L2'], help='main loss mode')
    parser.add_argument('--lossfun_alpha', type=float, default=0.0, help='alpha for time-weighted loss')
    parser.add_argument('--Q_loss', type=int, default=0, help='use Q_mat in loss function')
    parser.add_argument('--Q_loss_alpha', type=float, default=0.5, help='weight for Q_loss')
    parser.add_argument('--FFT_loss', type=int, default=0, help='use FFT_loss')
    parser.add_argument('--lamda1', type=float, default=1.0, help='weight for intermediate loss')
    parser.add_argument('--alpha', type=float, default=0.0, help='alpha for intermediate weights')
    parser.add_argument('--git_multi_stage', type=int, default=0, help='num of stages for intermediate loss')

    # =========================
    # 6. 测试与评估
    # =========================
    parser.add_argument('--test_batch_size', type=int, default=1, help='test_batch_size')
    parser.add_argument('--test_mode', type=int, default=0, help='0: normal test, other: specific test modes')
    parser.add_argument('--save_pdf', type=int, default=0, help='save prediction results as pdf')
    parser.add_argument('--inverse', type=int, default=0, help='inverse output data')
    parser.add_argument('--model_stats_mode', type=int, default=0, help='compute model statistics (FLOPS/Params)')
    parser.add_argument('--save_linear_weight', type=int, default=0, help='save linear weights')
    parser.add_argument('--save_linear_weight_path', type=str, default='weight_npy', help='path to save weights')
    parser.add_argument('--save_linear_weight_tag', type=str, default='tag', help='tag for saved weights')
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio for imputation task')
    parser.add_argument('--eval_flag', type=int, default=1, help='evaluation flag')

    # =========================
    # 7. GPU 设置
    # =========================
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

    # =========================
    # 8. 兼容参数
    # =========================
    parser.add_argument('--Patch_layer_num', type=int, default=0, help='Compat')
    parser.add_argument('--Patch_layer_num2', type=int, default=0, help='Compat')
    parser.add_argument('--second_e_layers', type=int, default=0, help='Compat')
    parser.add_argument('--decoder_cat_num', type=int, default=0, help='Compat')
    parser.add_argument('--seq_inter', type=int, default=0, help='Compat')
    parser.add_argument('--send_mail', type=int, default=0, help='send mail')
    parser.add_argument('--q_channel_file', type=str, default=None, help='Compat')
    parser.add_argument('--use_revin', type=int, default=1, help='Compat')
    parser.add_argument('--plot_mat_flag', type=int, default=0, help='Compat')
    parser.add_argument('--plot_mat_label', type=str, default='', help='Compat')

    # =========================
    # 10. D2GraNet 超参数
    # =========================
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--w_ratio', type=float, default=0.05, help='Sparsity ratio')
    parser.add_argument('--num_dynamic_clusters', type=int, default=15, help='Num dynamic clusters')
    parser.add_argument('--num_static_clusters', type=int, default=15, help='Num static clusters')
    parser.add_argument('--fcm_m', type=float, default=2.0, help='Fuzziness exponent m')
    parser.add_argument('--fcm_threshold', type=float, default=0.5, help='Threshold for FCM')

    parser.add_argument('--use_global_graph', type=int, default=1, help='Ablation: global')
    parser.add_argument('--use_hyper_graph', type=int, default=1, help='Ablation: hyper')
    parser.add_argument('--use_dynamic', type=int, default=1, help='Ablation: dynamic')
    parser.add_argument('--use_static_h', type=int, default=1, help='Ablation: static h')
    parser.add_argument('--use_ffn', type=int, default=1, help='Ablation: ffn')

    args = parser.parse_args()

    # =========================================================
    # 参数检查与环境设置
    # =========================================================
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print('args.use_gpu:', args.use_gpu)

    if args.train_ratio < 1:
        print(f'Only {args.train_ratio:.2%} of training set is used.')
    else:
        print(f'All training set is used.')

    if args.resume_training > 0 >= args.resume_epoch:
        confirm_again = input('args.resume_training > 0 >= args.resume_epoch. Continue? (yes/no):')
        if not confirm_again.lower().startswith('y'):
            sys.exit()

    if not args.use_gpu:
        confirm_again = input('No using gpu. Continue? (yes/no):')
        if not confirm_again.lower().startswith('y'):
            sys.exit()

    if (args.itr > 1 and args.fix_seed) or (args.itr == 1 and args.fix_seed == 0):
        # 原逻辑保留
        pass

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.task_name == 'imputation':
        args.pred_len = args.seq_len

    print('Args in experiment:')
    args_dict = vars(args)
    for k, v in sorted(args_dict.items()):
        print(f'{k}: {v}, ', end=' ')
    print('')

    # =========================================================
    # 实验初始化
    # =========================================================
    Exp = Exp_Forecast

    model_id_ori = args.model_id
    args.model_id_ori = model_id_ori
    args.model_id = model_id_ori + '_' + datetime.now().strftime('%y%m%d_%H%M%S')

    log_txt = 'log.txt'
    best_log_txt = 'best_log.txt'
    best_log_dataset_path = 'best_results'
    if not os.path.exists(best_log_dataset_path):
        os.makedirs(best_log_dataset_path)
    best_log_dataset_txt = os.path.join(best_log_dataset_path, model_id_ori + '.txt')

    test_batch_size_list = [args.test_batch_size]
    global_time0 = time.time()

    if args.fix_seed:
        fix_seed = 202601
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

    setting_zero = get_setting(args, 0)
    folder_path = os.path.join('results', setting_zero)
    args.folder_path = folder_path

    best_mse, best_mae = math.inf, math.inf
    time_vec = []

    # =========================================================
    # 执行模式逻辑 (Training / Testing / Stats)
    # =========================================================
    if args.is_training and not args.model_stats_mode and not args.save_linear_weight:
        test_batch_size_ori = args.test_batch_size
        best_ii = 0

        if test_batch_size_ori not in test_batch_size_list:
            test_batch_size_list.append(test_batch_size_ori)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if args.checkpoint_check and not args.resume_training and args.test_mode == 0:
            full_folder, new_setting = find_most_recently_modified_subfolder('./checkpoints/',
                                                                             file_name='checkpoint.pth',
                                                                             contain_str=args.model_id_ori)
            if full_folder is not None:
                print(f'{args.model_id_ori} checkpoints already exist.')
                sys.exit()

        idx = 0
        for ii in range(args.itr):
            time_now = time.time()
            idx += 1

            args.model_id = model_id_ori + '_' + datetime.now().strftime('%y%m%d_%H%M%S')
            setting = get_setting(args, ii) if ii > 0 else setting_zero

            if ii > 0:
                args.folder_path = os.path.join('results', setting)

            exp = Exp(args)

            # Training
            if args.test_mode == 0:
                print(f'>>>>>>>start training : {setting} >>')
                exp.train(setting)
            else:
                if ii > 0: break

            # Testing
            mse = mae = math.inf
            best_batch_size = np.nan
            for test_bs in sorted(test_batch_size_list):
                print(f'>>>>>>>testing : {setting} (test_batch_size: {test_bs})<<')
                mse0, mae0 = exp.test(setting, test=args.test_mode, test_batch_size=test_bs)
                if mse0 < mse:
                    mse, mae = mse0, mae0
                    best_batch_size = test_bs

            print(f'\tbest_test_batch_size: {best_batch_size}, best_mse: {mse:.5f}, best_mae: {mae:.5f}')

            if mse + mae <= best_mse + best_mae:
                best_mse, best_mae, best_ii = mse, mae, ii

            mse_mse_string = (f'mse:{mse:.5f}, mae:{mae:.5f}, lamda1:{args.lamda1:.2f}')
            print(mse_mse_string)

            # 建议：这里的 log_txt 可以改为 args.folder_path 下的日志，防止冲突
            # 但为了保持原功能，此处不强行修改
            with open(log_txt, 'a') as f:
                f.write(f'------------ {setting} -------------\n')
                f.write(f'\t{mse_mse_string}\n')
                f.write('--------------------------------- Ends -----------------------------\n')

            time_vec.append(time.time() - time_now)

    elif not args.model_stats_mode and not args.is_training and not args.save_linear_weight:
        ii = 0
        setting = get_setting(args, ii)
        exp = Exp(args)
        print(f'>>>>>>>testing : {setting}<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

    elif args.model_stats_mode and not args.save_linear_weight:
        print(f'=== model_stats {args.model_id_ori} ===')
        exp = Exp(args)
        exp.compute_model_stats()

    elif args.save_linear_weight:
        print(f'=== save_linear_weight {args.model_id_ori} ===')
        exp = Exp(args)
        exp.save_linear_weight_2npy(setting=setting_zero)

    print(f'A total of {(time.time() - global_time0) / 60.0: .2f} min(s) used...')

    if args.send_mail and args.is_training and args.test_mode == 0:
        mess_body = f'{args.task_name}_{args.model_id} program complete.'
        send_email(body=mess_body)