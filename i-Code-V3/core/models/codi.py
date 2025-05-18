from typing import Dict, List # 导入类型提示，用于代码可读性和静态分析
import os # 导入操作系统接口模块，可能用于文件路径等操作

import torch # 导入 PyTorch 核心库
import torch.nn as nn # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F # 导入 PyTorch 的神经网络函数库
import numpy as np # 导入 NumPy 库，用于数值计算
import numpy.random as npr # 导入 NumPy 的随机数生成模块
import copy # 导入 copy 模块，用于对象的深拷贝和浅拷贝
from functools import partial # 导入 functools 模块的 partial 函数，用于创建偏函数
from contextlib import contextmanager # 导入 contextlib 模块的 contextmanager 装饰器，用于创建上下文管理器

from .common.get_model import get_model, register # 从相对路径 .common.get_model 导入 get_model 和 register 函数
                                                # 这表明项目有一个模型注册和获取的机制
from .sd import DDPM # 从相对路径 .sd 导入 DDPM 类，这是 CoDi 类的基类

version = '0' # 定义版本号字符串
symbol = 'codi' # 定义模型符号字符串
    
    
@register('codi', version) # 使用 @register 装饰器将 CoDi 类注册到模型工厂中
                          # 这样可以通过 get_model('codi', version) 来获取 CoDi 类的实例
class CoDi(DDPM): # 定义 CoDi 类，继承自 DDPM 类
    def __init__(self, # 定义 CoDi 类的初始化方法
                 audioldm_cfg, # AudioLDM 模型的配置参数
                 autokl_cfg,   # AutoKL (Autoencoder KL) 模型的配置参数
                 optimus_cfg,  # Optimus (文本 VAE) 模型的配置参数
                 clip_cfg,     # CLIP 模型的配置参数
                 clap_cfg,     # CLAP 模型的配置参数
                 vision_scale_factor=0.1812, # 视觉特征的缩放因子，默认值
                 text_scale_factor=4.3108,   # 文本特征的缩放因子，默认值
                 audio_scale_factor=0.9228,  # 音频特征的缩放因子，默认值
                 scale_by_std=False, # 是否通过标准差来缩放特征，默认为 False
                 *args, # 接收任意数量的位置参数，传递给父类
                 **kwargs): # 接收任意数量的关键字参数，传递给父类
        super().__init__(*args, **kwargs) # 调用父类 DDPM 的初始化方法，传递参数
        
        # --- 初始化各个预训练的单模态模型 ---
        self.audioldm = get_model()(audioldm_cfg) # 获取并初始化 AudioLDM 模型实例
                                                 # get_model() 可能返回一个模型类，然后 ()audioldm_cfg 进行实例化
        
        self.autokl = get_model()(autokl_cfg) # 获取并初始化 AutoKL 模型实例
            
        self.optimus = get_model()(optimus_cfg) # 获取并初始化 Optimus 模型实例
            
        self.clip = get_model()(clip_cfg) # 获取并初始化 CLIP 模型实例
            
        self.clap = get_model()(clap_cfg) # 获取并初始化 CLAP 模型实例
        
        # --- 处理特征缩放因子 ---
        if not scale_by_std: # 如果不按标准差缩放 (即使用固定的缩放因子)
            self.vision_scale_factor = vision_scale_factor # 将 vision_scale_factor 存为实例属性
            self.text_scale_factor = text_scale_factor   # 将 text_scale_factor 存为实例属性
            self.audio_scale_factor = audio_scale_factor # 将 audio_scale_factor 存为实例属性
        else: # 如果按标准差缩放 (这些因子可能是在训练数据上计算得到的标准差的倒数)
            self.register_buffer("text_scale_factor", torch.tensor(text_scale_factor)) # 将 text_scale_factor 注册为 PyTorch buffer
            self.register_buffer("audio_scale_factor", torch.tensor(audio_scale_factor))# 将 audio_scale_factor 注册为 PyTorch buffer
            self.register_buffer('vision_scale_factor', torch.tensor(vision_scale_factor))# 将 vision_scale_factor 注册为 PyTorch buffer
            # register_buffer 使其成为模型状态的一部分，会被保存和加载，但不是可训练参数

        self.freeze() # 调用 freeze 方法冻结模型参数
        
    def freeze(self): # 定义冻结模型参数的方法
        self.eval() # 将模型设置为评估模式 (例如，关闭 Dropout, BatchNorm 使用运行均值和方差)
        for param in self.parameters(): # 遍历模型的所有参数 (包括父类 DDPM 和这里定义的各个子模型的参数)
            param.requires_grad = False # 将每个参数的 requires_grad 属性设置为 False，使其不可训练
            
    @property # 使用 @property 装饰器，使 device 方法可以像属性一样被访问 (例如 self.device)
    def device(self): # 获取模型所在的设备 (CPU 或 GPU)
        return next(self.parameters()).device # 返回模型第一个参数所在的设备
                                              # 这是一个常用的获取模型设备的方法
            
    @torch.no_grad() # 装饰器，表示该方法内的所有 PyTorch 操作都不会计算梯度
    def autokl_encode(self, image): # 定义 AutoKL 编码方法，用于编码图像
        # image: 输入的图像张量
        encoder_posterior = self.autokl.encode(image) # 调用 AutoKL 模型的 encode 方法获取编码器的后验分布
        z = encoder_posterior.sample().to(image.dtype) # 从后验分布中采样得到潜变量 z，并转换为与输入图像相同的数据类型
        return self.vision_scale_factor * z # 返回缩放后的潜变量 z

    @torch.no_grad() # 不计算梯度
    def autokl_decode(self, z): # 定义 AutoKL 解码方法，用于从潜变量解码图像
        # z: 输入的潜变量张量
        z = 1. / self.vision_scale_factor * z # 对潜变量进行反向缩放
        return self.autokl.decode(z) # 调用 AutoKL 模型的 decode 方法解码图像

    @torch.no_grad() # 不计算梯度
    def optimus_encode(self, text): # 定义 Optimus 编码方法，用于编码文本
        # text: 输入的文本，可以是字符串列表或预处理好的 token_id 张量
        if isinstance(text, List): # 如果输入是字符串列表
            tokenizer = self.optimus.tokenizer_encoder # 获取 Optimus 模型的编码器分词器
            token = [tokenizer.tokenize(sentence.lower()) for sentence in text] # 对每个句子进行小写转换和分词
            token_id = [] # 初始化 token_id 列表
            for tokeni in token: # 遍历每个分词后的句子
                token_sentence = [tokenizer._convert_token_to_id(i) for i in tokeni] # 将 token 转换为 token_id
                token_sentence = tokenizer.add_special_tokens_single_sentence(token_sentence) # 添加特殊标记 (如 [CLS], [SEP])
                token_id.append(torch.LongTensor(token_sentence)) # 将 token_id 列表转换为长整型张量并添加到列表中
            # 将 token_id 列表中的张量进行填充，使其长度一致，并截断到最大长度 512
            token_id = torch._C._nn.pad_sequence(token_id, batch_first=True, padding_value=0.0)[:, :512]
        else: # 如果输入已经是 token_id 张量
            token_id = text #直接使用
        # 使用 Optimus 编码器获取文本表示，[1] 通常取的是 [CLS] token 或者池化后的句子表示
        z = self.optimus.encoder(token_id, attention_mask=(token_id > 0))[1]
        # Optimus 是 VAE 结构，编码器输出均值 z_mu 和对数方差 z_logvar
        z_mu, z_logvar = self.optimus.encoder.linear(z).chunk(2, -1) # 通过线性层并切分得到均值和对数方差
        return z_mu.squeeze(1) * self.text_scale_factor # 返回缩放后的均值 z_mu (移除多余的维度)

    @torch.no_grad() # 不计算梯度
    def optimus_decode(self, z, temperature=1.0): # 定义 Optimus 解码方法，用于从潜变量解码文本
        # z: 输入的文本潜变量
        # temperature: 解码时控制生成多样性的温度参数
        z = 1.0 / self.text_scale_factor * z # 对潜变量进行反向缩放
        return self.optimus.decode(z, temperature) # 调用 Optimus 模型的 decode 方法解码文本

    @torch.no_grad() # 不计算梯度
    def audioldm_encode(self, audio, time=2.0): # 定义 AudioLDM 编码方法，用于编码音频
        # audio: 输入的音频波形
        # time: 音频时长（可能用于模型内部处理）
        encoder_posterior = self.audioldm.encode(audio, time=time) # 调用 AudioLDM 模型的 encode 方法获取编码器后验
        z = encoder_posterior.sample().to(audio.dtype) # 从后验采样得到潜变量，并转换为与输入音频相同的数据类型
        return z * self.audio_scale_factor # 返回缩放后的潜变量 z

    @torch.no_grad() # 不计算梯度
    def audioldm_decode(self, z): # 定义 AudioLDM 解码方法，用于从潜变量解码音频
        # z: 输入的音频潜变量
        if (torch.max(torch.abs(z)) > 1e2): # 如果潜变量的绝对值最大值过大
            z = torch.clip(z, min=-10, max=10) # 将潜变量裁剪到 [-10, 10] 范围内，防止数值不稳定
        z = 1.0 / self.audio_scale_factor * z # 对潜变量进行反向缩放
        return self.audioldm.decode(z) # 调用 AudioLDM 模型的 decode 方法解码音频

    @torch.no_grad() # 不计算梯度
    def mel_spectrogram_to_waveform(self, mel): # 将梅尔频谱图转换为音频波形的方法
        # mel: 输入的梅尔频谱图，期望形状 [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4: # 如果梅尔频谱图是4维的 (带有通道维度)
            mel = mel.squeeze(1) # 移除通道维度，变为 [bs, t-steps, fbins]
        mel = mel.permute(0, 2, 1) # 交换时间步和频率轴，变为 [bs, fbins, t-steps]，以匹配声码器输入
        waveform = self.audioldm.vocoder(mel) # 使用 AudioLDM 的声码器 (vocoder) 将梅尔频谱图转换为波形
        waveform = waveform.cpu().detach().numpy() # 将波形转移到 CPU，分离计算图，并转换为 NumPy 数组
        return waveform # 返回 NumPy 格式的音频波形

    @torch.no_grad() # 不计算梯度
    def clip_encode_text(self, text, encode_type='encode_text'): # 定义 CLIP 文本编码方法
        # text: 输入的文本
        # encode_type: 指定 CLIP 模型的编码类型，默认为 'encode_text'
        swap_type = self.clip.encode_type # 保存 CLIP 模型当前的编码类型
        self.clip.encode_type = encode_type # 设置 CLIP 模型的编码类型为指定的类型
        embedding = self.clip.encode(text) # 调用 CLIP 模型的 encode 方法获取文本嵌入
        self.clip.encode_type = swap_type # 恢复 CLIP 模型原始的编码类型
        return embedding # 返回文本嵌入

    @torch.no_grad() # 不计算梯度
    def clip_encode_vision(self, vision, encode_type='encode_vision'): # 定义 CLIP 视觉编码方法
        # vision: 输入的图像
        # encode_type: 指定 CLIP 模型的编码类型，默认为 'encode_vision'
        swap_type = self.clip.encode_type # 保存 CLIP 模型当前的编码类型
        self.clip.encode_type = encode_type # 设置 CLIP 模型的编码类型
        embedding = self.clip.encode(vision) # 调用 CLIP 模型的 encode 方法获取视觉嵌入
        self.clip.encode_type = swap_type # 恢复 CLIP 模型原始的编码类型
        return embedding # 返回视觉嵌入

    @torch.no_grad() # 不计算梯度
    def clap_encode_audio(self, audio): # 定义 CLAP 音频编码方法
        # audio: 输入的音频
        embedding = self.clap(audio) # 直接调用 CLAP 模型 (其 __call__ 方法) 获取音频嵌入
        return embedding # 返回音频嵌入

    def forward(self, x=None, c=None, noise=None, xtype='image', ctype='prompt', u=None, return_algined_latents=False):
        # CoDi 模型的主要前向传播方法，通常在训练时调用以计算损失
        # x: 目标生成的模态的起始数据 (x_start)，例如干净的图像、文本潜变量、音频潜变量
        # c: 条件信息，例如文本提示、图像条件、音频条件
        # noise: 可选的预定义噪声，如果为 None，则随机生成
        # xtype: x 的模态类型，例如 'image', 'text', 'audio', 'video'
        # ctype: c 的模态类型，例如 'prompt' (文本提示)
        # u: 可能的无条件指导 (unconditional guidance) 的条件，通常为 None
        # return_algined_latents: 是否返回对齐后的潜变量（用于特定分析或目的）
        
        if isinstance(x, list): # 如果 x_start 是一个列表 (表示多目标模态生成)
            # 为批次中的每个样本随机选择一个时间步 t (从 0 到 num_timesteps-1)
            t = torch.randint(0, self.num_timesteps, (x[0].shape[0],), device=x[0].device).long()
        else: # 如果 x_start 是单个张量
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        # 调用 p_losses 方法计算损失
        return self.p_losses(x, c, t, noise, xtype, ctype, u, return_algined_latents)

    def apply_model(self, x_noisy, t, cond, xtype='image', ctype='prompt', u=None, return_algined_latents=False):
        # 应用底层的扩散模型 (U-Net) 来预测噪声或 x0
        # x_noisy: 加噪后的数据 x_t
        # t: 当前时间步
        # cond: 条件信息
        # xtype, ctype, u, return_algined_latents: 与 forward 方法中的参数含义相同
        # self.model 属性通常指向包含 U-Net 的实际扩散模型实现 (可能是 LatentDiffusion 的一个实例)
        return self.model.diffusion_model(x_noisy, t, cond, xtype, ctype, u, return_algined_latents)

    def get_pixel_loss(self, pred, target, mean=True): # 计算基于像素的损失 (用于图像、视频、音频频谱)
        # pred: 模型预测值
        # target: 真实目标值
        # mean: 是否计算均值损失
        if self.loss_type == 'l1': # 如果损失类型是 L1 损失
            loss = (target - pred).abs() # 计算绝对差值
            if mean: # 如果需要计算均值
                loss = loss.mean() # 计算所有元素的均值
        elif self.loss_type == 'l2': # 如果损失类型是 L2 损失 (均方误差)
            if mean: # 如果需要计算均值
                loss = torch.nn.functional.mse_loss(target, pred) # 计算均方误差
            else: # 如果不需要计算均值 (返回每个元素的损失)
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none') # 计算逐元素的均方误差
        else: # 其他未实现的损失类型
            raise NotImplementedError("unknown loss type '{loss_type}'")
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=-0.0) # 将 NaN 和 Inf 值替换为 0.0，防止训练崩溃
        return loss # 返回计算得到的损失

    def get_text_loss(self, pred, target): # 计算文本潜变量的损失
        # pred: 模型预测的文本潜变量
        # target: 真实的文本潜变量
        if self.loss_type == 'l1': # L1 损失
            loss = (target - pred).abs()
        elif self.loss_type == 'l2': # L2 损失 (逐元素)
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0) # NaN 和 Inf 处理   
        return loss # 返回损失

    def p_losses(self, x_start, cond, t, noise=None, xtype='image', ctype='prompt', u=None, return_algined_latents=False):
        # 计算扩散模型的训练损失 (p_losses 通常指代 reverse process 的损失)
        # 参数含义与 forward 方法中的类似
        
        if isinstance(x_start, list): # 如果是多目标模态生成 (x_start 是一个列表)
            # 如果 noise 未提供，则为每个 x_start_i 生成对应形状的随机高斯噪声
            noise = [torch.randn_like(x_start_i) for x_start_i in x_start] if noise is None else noise
            # 对每个 x_start_i 应用 q_sample (前向扩散过程) 得到加噪后的 x_noisy_i
            x_noisy = [self.q_sample(x_start=x_start_i, t=t, noise=noise_i) for x_start_i, noise_i in zip(x_start, noise)]
            # 应用扩散模型 (U-Net) 得到预测结果 (可能是预测的噪声 eps 或预测的 x0)
            # 注意：这里的 xtype 应该也是一个列表，对应每个 x_noisy_i 的模态
            model_output = self.apply_model(x_noisy, t, cond, xtype, ctype, u, return_algined_latents)
            
            if return_algined_latents: # 如果只需要返回对齐后的潜变量
                return model_output # 直接返回
            
            loss_dict = {} # 初始化损失字典 (虽然这里没用到)

            # 根据参数化类型确定目标 (是预测 x0 还是预测噪声 eps)
            if self.parameterization == "x0":
                target = x_start # 如果预测 x0，目标就是原始的 x_start
            elif self.parameterization == "eps":
                target = noise # 如果预测噪声，目标就是加入的噪声
            else:
                raise NotImplementedError()
            
            loss = 0.0 # 初始化总损失
            # 遍历每个模态的预测输出、目标和模态类型
            for model_output_i, target_i, xtype_i in zip(model_output, target, xtype):
                if xtype_i == 'image': # 如果是图像模态
                    # 计算像素损失，并在通道、高、宽维度上取均值，得到每个样本的损失
                    loss_simple = self.get_pixel_loss(model_output_i, target_i, mean=False).mean([1, 2, 3])
                elif xtype_i == 'video': # 如果是视频模态 (通常视频帧也被视为图像)
                    # 计算像素损失，并在通道、时间、高、宽维度上取均值
                    loss_simple = self.get_pixel_loss(model_output_i, target_i, mean=False).mean([1, 2, 3, 4])
                elif xtype_i == 'text': # 如果是文本模态 (通常指文本的潜变量表示)
                    # 计算文本潜变量损失，并在潜变量维度上取均值
                    loss_simple = self.get_text_loss(model_output_i, target_i).mean([1])
                elif xtype_i == 'audio': # 如果是音频模态 (通常指音频的频谱图或潜变量表示)
                    # 计算像素损失 (频谱图可视为图像)，并在通道、频率、时间维度上取均值
                    loss_simple = self.get_pixel_loss(model_output_i, target_i, mean=False).mean([1, 2, 3])
                loss += loss_simple.mean() # 将当前模态的平均损失累加到总损失中
            return loss / len(xtype) # 返回所有模态的平均损失
        
        else: # 如果是单目标模态生成
            # 如果 noise 未提供，则生成与 x_start 形状相同的随机高斯噪声
            noise = torch.randn_like(x_start) if noise is None else noise
            # 应用 q_sample (前向扩散过程) 得到加噪后的 x_noisy
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            # 应用扩散模型 (U-Net) 得到预测结果
            model_output = self.apply_model(x_noisy, t, cond, xtype, ctype)

            loss_dict = {} # 初始化损失字典 (虽然这里没用到)

            # 根据参数化类型确定目标
            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = noise
            else:
                raise NotImplementedError()

            # 根据模态类型计算损失
            if xtype == 'image':
                loss_simple = self.get_pixel_loss(model_output, target, mean=False).mean([1, 2, 3])
            elif xtype == 'video':
                loss_simple = self.get_pixel_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
            elif xtype == 'text':
                loss_simple = self.get_text_loss(model_output, target).mean([1])
            elif xtype == 'audio':
                loss_simple = self.get_pixel_loss(model_output, target, mean=False).mean([1, 2, 3])
            loss = loss_simple.mean() # 计算批次中所有样本的平均损失
            return loss # 返回损失