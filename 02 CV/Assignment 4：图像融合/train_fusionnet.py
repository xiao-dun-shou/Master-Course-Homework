# Training a NestFuse network
# auto-encoder

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import NestFuse_light2_nodense, Fusion_network
from args_fusion import args
import pytorch_msssim

EPSILON = 1e-5


def main():
	#获取参数路径下所有的图像数据绝对路径
	original_imgs_path, _ = utils.list_images(args.dataset_ir)

	#设置训练数据的数量，并打乱顺序
	train_num = 80000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)

	# True - RGB , False - gray
	#设置图像数据读取为灰度图
	img_flag = False

	#设置目标函数中L_detail的权重和\Phi_vi^m与\Phi_ir^m的权重
	alpha_list = [700]
	w_all_list = [[6.0, 3.0]]

	#根据不同的L_detail的权重进行训练
	for w_w in w_all_list:
		w1, w2 = w_w
		for alpha in alpha_list:
			train(original_imgs_path, img_flag, alpha, w1, w2)


def train(original_imgs_path, img_flag, alpha, w1, w2):

	#设置batch大小、输入输出通道、融合模块通道数和类型
	batch_size = args.batch_size
	nc = 1
	input_nc = nc
	output_nc = nc
	nb_filter = [64, 112, 160, 208, 256]
	f_type = 'res'

	#根据参数构造编码器、解码器模块并加载参数
	with torch.no_grad():
		deepsupervision = False
		nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision)
		model_path = args.resume_nestfuse
		# load auto-encoder network
		print('Resuming, initializing auto-encoder using weight from {}.'.format(model_path))
		nest_model.load_state_dict(torch.load(model_path))
		nest_model.eval()

	#根据参数构造融合模块并加载参数
	fusion_model = Fusion_network(nb_filter, f_type)
	if args.resume_fusion_model is not None:
		print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_fusion_model))
		fusion_model.load_state_dict(torch.load(args.resume_fusion_model))

	#定义优化器并加载上次训练时的参数
	optimizer = Adam(fusion_model.parameters(), args.lr)

	#定义损失函数
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	#是否使用GPU
	if args.cuda:
		nest_model.cuda()
		fusion_model.cuda()

	#确定训练epoch次数
	tbar = trange(args.epochs)
	print('Start training.....')

	#创建一些保存训练数据的路径，auto-encoder与残差融合模块分开
	temp_path_model = os.path.join(args.save_fusion_model)
	temp_path_loss  = os.path.join(args.save_loss_dir)
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)
	temp_path_model_w = os.path.join(args.save_fusion_model, str(w1))
	temp_path_loss_w  = os.path.join(args.save_loss_dir, str(w1))
	if os.path.exists(temp_path_model_w) is False:
		os.mkdir(temp_path_model_w)
	if os.path.exists(temp_path_loss_w) is False:
		os.mkdir(temp_path_loss_w)

	#设置变量存储训练过程中的loss
	Loss_feature = []
	Loss_ssim = []
	Loss_all = []
	count_loss = 0
	all_ssim_loss = 0.
	all_fea_loss = 0.
	#开始迭代epoch
	for e in tbar:
		print('Epoch %d.....' % e)
		#确定训练数据绝对路径，以及共有多少个batch，去掉冗余数据
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		fusion_model.train()
		fusion_model.cuda()

		count = 0

		for batch in range(batches):
			#读取当前batch的图像路径，然后读取数据同时resize
			image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img_ir = utils.get_train_images(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, flag=img_flag)

			#将前面读取数据路径中的lwir字符串改为visible，读取对应的可见光图像数据
			#前面确定的数据路径都是红外图像数据
			image_paths_vi = [x.replace('lwir', 'visible') for x in image_paths_ir]
			img_vi = utils.get_train_images(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, flag=img_flag)

			#清梯度
			count += 1
			optimizer.zero_grad()

			#数据也是否使用GPU，与前面模型同步
			img_ir = Variable(img_ir, requires_grad=False)
			img_vi = Variable(img_vi, requires_grad=False)
			if args.cuda:
				img_ir = img_ir.cuda()
				img_vi = img_vi.cuda()

			#红外图像与可见光图像分别经过encoder提取特征
			en_ir = nest_model.encoder(img_ir)
			en_vi = nest_model.encoder(img_vi)
			#红外图像与可见光图像多尺度特征融合
			f = fusion_model(en_ir, en_vi)
			#重构融合图像
			outputs = nest_model.decoder_eval(f)

			#将GPU显存中的两个输入图像数据复制到CPU，用于计算LOSS
			x_ir = Variable(img_ir.data.clone(), requires_grad=False)
			x_vi = Variable(img_vi.data.clone(), requires_grad=False)

			######################### LOSS FUNCTION #########################
			loss1_value = 0.
			loss2_value = 0.
			#batch中各个重构图像分别计算loss
			for output in outputs:
				#这里实现重构图像的归一化，然后限制范围到[0,255]
				output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
				output = output * 255
				# ---------------------- LOSS IMAGES ------------------------------------

				#计算detail loss
				# ssim_loss_temp1 = ssim_loss(output, x_ir, normalize=True)
				ssim_loss_temp2 = ssim_loss(output, x_vi, normalize=True)
				loss1_value += alpha * (1 - ssim_loss_temp2)

				#计算feature loss
				g2_ir_fea = en_ir
				g2_vi_fea = en_vi
				g2_fuse_fea = f
				# w_ir = [3.5, 3.5, 3.5, 3.5]
				w_ir = [w1, w1, w1, w1]
				w_vi = [w2, w2, w2, w2]
				w_fea = [1, 10, 100, 1000]
				for ii in range(4):
					g2_ir_temp = g2_ir_fea[ii]
					g2_vi_temp = g2_vi_fea[ii]
					g2_fuse_temp = g2_fuse_fea[ii]
					(bt, cht, ht, wt) = g2_ir_temp.size()
					loss2_value += w_fea[ii]*mse_loss(g2_fuse_temp, w_ir[ii]*g2_ir_temp + w_vi[ii]*g2_vi_temp)

			#取batch内的loss均值
			loss1_value /= len(outputs)
			loss2_value /= len(outputs)

			#更新参数
			total_loss = loss1_value + loss2_value
			total_loss.backward()
			optimizer.step()

			all_fea_loss += loss2_value.item() # 
			all_ssim_loss += loss1_value.item() # 
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\t Alpha: {} \tW-IR: {}\tEpoch {}:\t[{}/{}]\t ssim loss: {:.6f}\t fea loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), alpha, w1, e + 1, count, batches,
								  all_ssim_loss / args.log_interval,
								  all_fea_loss / args.log_interval,
								  (all_fea_loss + all_ssim_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_ssim.append( all_ssim_loss / args.log_interval)
				Loss_feature.append(all_fea_loss / args.log_interval)
				Loss_all.append((all_fea_loss + all_ssim_loss) / args.log_interval)
				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_fea_loss = 0.

			#每隔固定训练batch就保存loss
			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model
				fusion_model.eval()
				fusion_model.cpu()

				save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".model"
				save_model_path = os.path.join(temp_path_model, save_model_filename)
				torch.save(fusion_model.state_dict(), save_model_path)
				# save loss data
				# pixel loss
				loss_data_ssim = Loss_ssim
				loss_filename_path = temp_path_loss_w + "/loss_ssim_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
				# SSIM loss
				loss_data_fea = Loss_feature
				loss_filename_path = temp_path_loss_w + "/loss_fea_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				scio.savemat(loss_filename_path, {'loss_fea': loss_data_fea})
				# all loss
				loss_data = Loss_all
				loss_filename_path = temp_path_loss_w + "/loss_all_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				scio.savemat(loss_filename_path, {'loss_all': loss_data})

				fusion_model.train()
				fusion_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

		#每隔固定训练batch就保存模型和损失
		loss_data_ssim = Loss_ssim
		loss_filename_path = temp_path_loss_w + "/Final_loss_ssim_epoch_" + str(
			args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
		scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
		loss_data_fea = Loss_feature
		loss_filename_path = temp_path_loss_w + "/Final_loss_2_epoch_" + str(
			args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
		scio.savemat(loss_filename_path, {'final_loss_fea': loss_data_fea})
		# SSIM loss
		loss_data = Loss_all
		loss_filename_path = temp_path_loss_w + "/Final_loss_all_epoch_" + str(
			args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
		scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
		# save model
		fusion_model.eval()
		fusion_model.cpu()
		save_model_filename = "Final_epoch_" + str(args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(
			w1) + "_wvi_" + str(w2) + ".model"
		save_model_path = os.path.join(temp_path_model_w, save_model_filename)
		torch.save(fusion_model.state_dict(), save_model_path)

		print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)


if __name__ == "__main__":
	main()
