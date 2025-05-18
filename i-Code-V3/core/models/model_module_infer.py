 import os # 导入操作系统接口模块，用于文件路径操作

import torch # 导入 PyTorch 核心库
import torch.nn as nn # 导入 PyTorch 神经网络模块
import torch.nn.functional as F # 导入 PyTorch 神经网络函数库
import torchvision.transforms as tvtrans # 导入 torchvision 的 transforms 模块，用于图像预处理和转换

from einops import rearrange # 导入 einops 库的 rearrange 函数，一个强大的张量操作工具

import pytorch_lightning as pl # 导入 PyTorch Lightning 库，一个 PyTorch 的高级封装，简化训练和推理流程

from . import get_model # 从当前目录的 __init__.py (或同级模块) 导入 get_model 函数
from ..cfg_helper import model_cfg_bank # 从上一级目录的 cfg_helper 模块导入 model_cfg_bank 类，用于获取模型配置
from ..common.utils import regularize_image, regularize_video, remove_duplicate_word # 从上一级目录的 common.utils 导入一些工具函数

import warnings # 导入 Python 的警告处理模块
warnings.filterwarnings("ignore") # 忽略所有警告信息，这在演示或库发布时可能不推荐，但开发时可能为了清爽输出


class model_module(pl.LightningModule): # 定义一个名为 model_module 的类，继承自 PyTorch Lightning 的 LightningModule
    def __init__(self, data_dir='pretrained', pth=["CoDi_encoders.pth"], fp16=False):
        # 初始化方法
        # data_dir: 存放预训练权重文件的目录，默认为 'pretrained'
        # pth: 一个包含预训练权重文件名的列表，默认为 ["CoDi_encoders.pth"]
        #      注意：这里的文件名 CoDi_encoders.pth 可能是一个包含了所有 CoDi 编码器和 "Bridge" 权重的融合文件，
        #      或者是一个包含了 CoDi 整体（包括扩散 U-Net）的权重文件。
        #      与之前讨论的 CoDi_text_diffuser.pth 等文件不同，后者更像是独立的解码器组件。
        # fp16: 是否使用半精度浮点数 (float16) 进行推理，默认为 False

        super().__init__() # 调用父类 pl.LightningModule 的初始化方法
        
        cfgm = model_cfg_bank()('codi') # 从模型配置库 model_cfg_bank 中获取 'codi' 模型的配置
        net = get_model()(cfgm) # 使用 get_model 函数和配置 cfgm 来创建 CoDi 网络模型实例 (可能是前一个代码块中的 CoDi 类或其封装)
        
        if fp16: # 如果启用半精度
            net = net.half() # 将网络模型转换为半精度
            
        for path in pth: # 遍历权重文件列表
            # 加载预训练权重到网络模型中
            # os.path.join(data_dir, path) 构造权重文件的完整路径
            # map_location='cpu' 确保权重首先加载到 CPU，避免 GPU 内存不足或设备不匹配问题
            # strict=False 允许加载部分匹配的权重，如果模型结构与权重文件不完全一致，不会报错
            net.load_state_dict(torch.load(os.path.join(data_dir, path), map_location='cpu'), strict=False)
        print('Load pretrained weight from {}'.format(pth)) # 打印加载权重的信息

        self.net = net # 将加载了权重的网络模型保存为实例属性 self.net
        
        from core.models.ddim.ddim_vd import DDIMSampler_VD # 从指定路径导入 DDIMSampler_VD 类
                                                          # DDIM (Denoising Diffusion Implicit Models) 是一种加速扩散模型采样的算法
                                                          # _VD 可能表示 Video Diffusion 或 Variable Dimensions
        self.sampler = DDIMSampler_VD(net) # 创建 DDIM 采样器实例，并将 CoDi 网络模型 net 传递给它

    def decode(self, z, xtype): # 定义解码方法，用于从潜变量 z 生成特定模态 xtype 的数据
        # z: 输入的潜变量张量
        # xtype: 目标输出的模态类型 ('image', 'video', 'text', 'audio')
        net = self.net # 获取 CoDi 网络模型
        z = z.cuda() # 将潜变量张量移动到 CUDA 设备 (GPU)
        
        if xtype == 'image': # 如果目标是图像
            x = net.autokl_decode(z) # 使用网络的 autokl_decode 方法从潜变量解码图像
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0) # 将解码后的图像数据归一化到 [0, 1] 范围
                                                         # AutoKL 输出通常在 [-1, 1]
            x = [tvtrans.ToPILImage()(xi) for xi in x] # 将批次中的每个图像张量转换为 PIL.Image 对象
            return x # 返回 PIL.Image 对象列表
        
        elif xtype == 'video': # 如果目标是视频
            num_frames = z.shape[2] # 获取潜变量 z 中帧的数量 (假设 z 的形状是 [b, c, f, h, w])
            # 使用 einops.rearrange 将视频潜变量的形状从 [b, c, f, h, w] 变为 [(b*f), c, h, w]
            # 这样可以将视频的每一帧视为一个独立的图像，方便使用图像解码器
            z = rearrange(z, 'b c f h w -> (b f) c h w')
            x = net.autokl_decode(z) # 使用图像解码器解码每一帧
            # 使用 einops.rearrange 将解码后的帧重新组合成视频形状 [(b*f), c, h, w] -> [b, f, c, h, w]
            x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
            
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0) # 归一化到 [0, 1]
            video_list = [] # 初始化视频列表
            for video in x: # 遍历批次中的每个视频
                # 将视频中的每一帧转换为 PIL.Image 对象
                video_list.append([tvtrans.ToPILImage()(xi) for xi in video])
            return video_list # 返回包含 PIL.Image 对象列表的列表 (每个子列表代表一个视频的帧)

        elif xtype == 'text': # 如果目标是文本
            prompt_temperature = 1.0 # 设置文本生成时的温度参数
            prompt_merge_same_adj_word = True # 是否合并相邻的相同单词
            x = net.optimus_decode(z, temperature=prompt_temperature) # 使用网络的 optimus_decode 方法解码文本
            if prompt_merge_same_adj_word: # 如果需要合并相邻重复词
                xnew = [] # 初始化新文本列表
                for xi in x: # 遍历生成的每个文本字符串
                    xi_split = xi.split() # 按空格分割单词
                    xinew = [] # 初始化处理后的单词列表
                    for idxi, wi in enumerate(xi_split): # 遍历单词
                        if idxi!=0 and wi==xi_split[idxi-1]: # 如果当前单词与前一个单词相同 (且不是第一个词)
                            continue # 跳过该重复单词
                        xinew.append(wi) # 添加不重复的单词
                    xnew.append(remove_duplicate_word(' '.join(xinew))) # 使用 remove_duplicate_word 进一步去重并加入列表
                                                                        # (这个函数可能处理更复杂的重复情况)
                x = xnew # 更新 x 为处理后的文本列表
            return x # 返回文本字符串列表
        
        elif xtype == 'audio': # 如果目标是音频
            x = net.audioldm_decode(z) # 使用网络的 audioldm_decode 方法从潜变量解码音频（可能是梅尔频谱图）
            x = net.mel_spectrogram_to_waveform(x) # 将解码得到的梅尔频谱图转换为音频波形
            return x # 返回音频波形 (NumPy 数组)

    def inference(self, xtype=[], condition=[], condition_types=[], n_samples=1, mix_weight={'video': 1, 'audio': 1, 'text': 1, 'image': 1}, image_size=256, ddim_steps=50, scale=7.5, num_frames=8):
        # 定义推理（生成）方法
        # xtype: 一个列表，包含希望生成的目标模态类型，例如 ['image', 'audio']
        # condition: 一个列表，包含输入的条件数据，例如 [PIL.Image 对象, "一段文本", audio_waveform]
        # condition_types: 一个列表，对应 condition 中每个条件的模态类型，例如 ['image', 'text', 'audio']
        # n_samples: 每个目标模态要生成的样本数量
        # mix_weight: (未使用在此片段的采样中，但可能用于更复杂的条件混合) 不同模态条件的混合权重
        # image_size: 生成图像的目标尺寸
        # ddim_steps: DDIM 采样的步数
        # scale: 无条件指导 (classifier-free guidance) 的尺度因子，控制条件对生成结果的影响程度
        # num_frames: 生成视频时的帧数

        net = self.net # 获取 CoDi 网络模型
        sampler = self.sampler # 获取 DDIM 采样器
        ddim_eta = 0.0 # DDIM 采样参数 eta，0.0 表示 DDIM (确定性)，1.0 表示 DDPM (随机性)

        conditioning = [] # 初始化条件嵌入列表，用于存储所有输入条件的编码结果
        # 断言：确保条件类型中没有重复的模态，当前实现可能不支持同一模态的多个条件
        assert len(set(condition_types)) == len(condition_types), "we don't support condition with same modalities yet."
        # 断言：确保条件数据和条件类型列表长度一致
        assert len(condition) == len(condition_types)
        
        # --- 对所有输入条件进行编码 ---
        for i, condition_type in enumerate(condition_types): # 遍历输入的每个条件及其类型
            if condition_type == 'image': # 如果条件是图像
                ctemp1 = regularize_image(condition[i]).cuda() # 对图像进行预处理 (如缩放、归一化) 并移到 GPU
                ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1) # 增加批次维度并复制 n_samples 次
                cim = net.clip_encode_vision(ctemp1).cuda() # 使用 CLIP 编码图像条件
                uim = None # 初始化无条件嵌入
                if scale != 1.0: # 如果使用 classifier-free guidance (scale > 1.0)
                    dummy = torch.zeros_like(ctemp1).cuda() # 创建一个与图像条件形状相同的零张量作为无条件输入
                    uim = net.clip_encode_vision(dummy).cuda() # 编码零张量得到无条件嵌入
                conditioning.append(torch.cat([uim, cim])) # 将无条件嵌入和条件嵌入拼接后加入列表
                                                          # (用于 classifier-free guidance)
                
            elif condition_type == 'video': # 如果条件是视频
                ctemp1 = regularize_video(condition[i]).cuda() # 对视频进行预处理并移到 GPU
                ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1, 1) # 增加批次维度并复制
                cim = net.clip_encode_vision(ctemp1).cuda() # 使用 CLIP (可能是处理视频帧序列的变体) 编码视频条件
                uim = None
                if scale != 1.0:
                    dummy = torch.zeros_like(ctemp1).cuda()
                    uim = net.clip_encode_vision(dummy).cuda()
                conditioning.append(torch.cat([uim, cim]))
                
            elif condition_type == 'audio': # 如果条件是音频
                ctemp = condition[i][None].repeat(n_samples, 1, 1) # 增加批次维度并复制 (假设输入音频形状 [T, F] 或 [L])
                cad = net.clap_encode_audio(ctemp) # 使用 CLAP 编码音频条件
                uad = None
                if scale != 1.0:
                    dummy = torch.zeros_like(ctemp)
                    uad = net.clap_encode_audio(dummy)  
                conditioning.append(torch.cat([uad, cad]))
                
            elif condition_type == 'text': # 如果条件是文本
                # 将文本字符串复制 n_samples 次，然后使用 CLIP 编码
                ctx = net.clip_encode_text(n_samples * [condition[i]]).cuda()
                utx = None
                if scale != 1.0:
                    # 使用空字符串作为无条件文本输入
                    utx = net.clip_encode_text(n_samples * [""]).cuda()
                conditioning.append(torch.cat([utx, ctx]))
        
        # --- 准备目标输出模态的潜变量形状 ---
        shapes = [] # 初始化形状列表
        for xtype_i in xtype: # 遍历希望生成的目标模态
            if xtype_i == 'image': # 如果目标是图像
                h, w = [image_size, image_size] # 获取图像高宽
                shape = [n_samples, 4, h//8, w//8] # 定义图像潜变量的形状
                                                   # 4 通常是 AutoKL 的潜空间通道数
                                                   # h//8, w//8 是因为 AutoKL 通常有 8 倍的下采样
            elif xtype_i == 'video': # 如果目标是视频
                h, w = [image_size, image_size]
                shape = [n_samples, 4, num_frames, h//8, w//8] # 定义视频潜变量的形状 (增加帧维度)
            elif xtype_i == 'text': # 如果目标是文本
                n = 768 # 文本潜变量的维度 (例如 Optimus 或 CLIP 文本特征维度)
                shape = [n_samples, n]
            elif xtype_i == 'audio': # 如果目标是音频
                h, w = [256, 16] # 音频潜变量（频谱图）的典型高宽 (频率 x 时间块)
                                 # 这些值取决于 AudioLDM 的具体配置
                shape = [n_samples, 8, h, w] # 定义音频潜变量的形状 (8 是潜空间通道数)
            else:
                raise # 如果是不支持的目标类型，则抛出异常
            shapes.append(shape) # 将形状加入列表
        

        # --- 使用 DDIM 采样器从噪声生成潜变量 z ---
        z, _ = sampler.sample( # 调用采样器的 sample 方法
            steps=ddim_steps, # DDIM 步数
            shape=shapes, # 目标潜变量形状列表 (对应每个 xtype_i)
            condition=conditioning, # 编码后的条件嵌入列表
            unconditional_guidance_scale=scale, # classifier-free guidance 尺度
            xtype=xtype, # 目标输出模态类型列表
            condition_types=condition_types, # 输入条件模态类型列表
            eta=ddim_eta, # DDIM eta 参数
            verbose=False, # 是否打印详细采样过程
            mix_weight=mix_weight # (可能未使用或用于更复杂的采样策略)
        )
        # z 返回的是一个潜变量列表，每个元素对应一个 xtype_i

        out_all = [] # 初始化所有输出结果的列表
        for i, xtype_i in enumerate(xtype): # 遍历每个目标模态及其生成的潜变量
            z[i] = z[i].cuda() # 将当前模态的潜变量移到 GPU
            x_i = self.decode(z[i], xtype_i) # 调用 decode 方法从潜变量解码得到具体数据
            out_all.append(x_i) # 将解码后的结果加入列表
        return out_all # 返回所有生成结果的列表 as pl # 导入 PyTorch Lightning 库，一个 PyTorch 的高级封装，简化训练和推理流程

from . import get_model # 从当前目录的 __init__.py (或同级模块) 导入 get_model 函数
from ..cfg_helper import model_cfg_bank # 从上一级目录的 cfg_helper 模块导入 model_cfg_bank 类，用于获取模型配置
from ..common.utils import regularize_image, regularize_video, remove_duplicate_word # 从上一级目录的 common.utils 导入一些工具函数

import warnings # 导入 Python 的警告处理模块
warnings.filterwarnings("ignore") # 忽略所有警告信息，这在演示或库发布时可能不推荐，但开发时可能为了清爽输出


class model_module(pl.LightningModule): # 定义一个名为 model_module 的类，继承自 PyTorch Lightning 的 LightningModule
    def __init__(self, data_dir='pretrained', pth=["CoDi_encoders.pth"], fp16=False):
        # 初始化方法
        # data_dir: 存放预训练权重文件的目录，默认为 'pretrained'
        # pth: 一个包含预训练权重文件名的列表，默认为 ["CoDi_encoders.pth"]
        #      注意：这里的文件名 CoDi_encoders.pth 可能是一个包含了所有 CoDi 编码器和 "Bridge" 权重的融合文件，
        #      或者是一个包含了 CoDi 整体（包括扩散 U-Net）的权重文件。
        #      与之前讨论的 CoDi_text_diffuser.pth 等文件不同，后者更像是独立的解码器组件。
        # fp16: 是否使用半精度浮点数 (float16) 进行推理，默认为 False

        super().__init__() # 调用父类 pl.LightningModule 的初始化方法
        
        cfgm = model_cfg_bank()('codi') # 从模型配置库 model_cfg_bank 中获取 'codi' 模型的配置
        net = get_model()(cfgm) # 使用 get_model 函数和配置 cfgm 来创建 CoDi 网络模型实例 (可能是前一个代码块中的 CoDi 类或其封装)
        
        if fp16: # 如果启用半精度
            net = net.half() # 将网络模型转换为半精度
            
        for path in pth: # 遍历权重文件列表
            # 加载预训练权重到网络模型中
            # os.path.join(data_dir, path) 构造权重文件的完整路径
            # map_location='cpu' 确保权重首先加载到 CPU，避免 GPU 内存不足或设备不匹配问题
            # strict=False 允许加载部分匹配的权重，如果模型结构与权重文件不完全一致，不会报错
            net.load_state_dict(torch.load(os.path.join(data_dir, path), map_location='cpu'), strict=False)
        print('Load pretrained weight from {}'.format(pth)) # 打印加载权重的信息

        self.net = net # 将加载了权重的网络模型保存为实例属性 self.net
        
        from core.models.ddim.ddim_vd import DDIMSampler_VD # 从指定路径导入 DDIMSampler_VD 类
                                                          # DDIM (Denoising Diffusion Implicit Models) 是一种加速扩散模型采样的算法
                                                          # _VD 可能表示 Video Diffusion 或 Variable Dimensions
        self.sampler = DDIMSampler_VD(net) # 创建 DDIM 采样器实例，并将 CoDi 网络模型 net 传递给它

    def decode(self, z, xtype): # 定义解码方法，用于从潜变量 z 生成特定模态 xtype 的数据
        # z: 输入的潜变量张量
        # xtype: 目标输出的模态类型 ('image', 'video', 'text', 'audio')
        net = self.net # 获取 CoDi 网络模型
        z = z.cuda() # 将潜变量张量移动到 CUDA 设备 (GPU)
        
        if xtype == 'image': # 如果目标是图像
            x = net.autokl_decode(z) # 使用网络的 autokl_decode 方法从潜变量解码图像
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0) # 将解码后的图像数据归一化到 [0, 1] 范围
                                                         # AutoKL 输出通常在 [-1, 1]
            x = [tvtrans.ToPILImage()(xi) for xi in x] # 将批次中的每个图像张量转换为 PIL.Image 对象
            return x # 返回 PIL.Image 对象列表
        
        elif xtype == 'video': # 如果目标是视频
            num_frames = z.shape[2] # 获取潜变量 z 中帧的数量 (假设 z 的形状是 [b, c, f, h, w])
            # 使用 einops.rearrange 将视频潜变量的形状从 [b, c, f, h, w] 变为 [(b*f), c, h, w]
            # 这样可以将视频的每一帧视为一个独立的图像，方便使用图像解码器
            z = rearrange(z, 'b c f h w -> (b f) c h w')
            x = net.autokl_decode(z) # 使用图像解码器解码每一帧
            # 使用 einops.rearrange 将解码后的帧重新组合成视频形状 [(b*f), c, h, w] -> [b, f, c, h, w]
            x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
            
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0) # 归一化到 [0, 1]
            video_list = [] # 初始化视频列表
            for video in x: # 遍历批次中的每个视频
                # 将视频中的每一帧转换为 PIL.Image 对象
                video_list.append([tvtrans.ToPILImage()(xi) for xi in video])
            return video_list # 返回包含 PIL.Image 对象列表的列表 (每个子列表代表一个视频的帧)

        elif xtype == 'text': # 如果目标是文本
            prompt_temperature = 1.0 # 设置文本生成时的温度参数
            prompt_merge_same_adj_word = True # 是否合并相邻的相同单词
            x = net.optimus_decode(z, temperature=prompt_temperature) # 使用网络的 optimus_decode 方法解码文本
            if prompt_merge_same_adj_word: # 如果需要合并相邻重复词
                xnew = [] # 初始化新文本列表
                for xi in x: # 遍历生成的每个文本字符串
                    xi_split = xi.split() # 按空格分割单词
                    xinew = [] # 初始化处理后的单词列表
                    for idxi, wi in enumerate(xi_split): # 遍历单词
                        if idxi!=0 and wi==xi_split[idxi-1]: # 如果当前单词与前一个单词相同 (且不是第一个词)
                            continue # 跳过该重复单词
                        xinew.append(wi) # 添加不重复的单词
                    xnew.append(remove_duplicate_word(' '.join(xinew))) # 使用 remove_duplicate_word 进一步去重并加入列表
                                                                        # (这个函数可能处理更复杂的重复情况)
                x = xnew # 更新 x 为处理后的文本列表
            return x # 返回文本字符串列表
        
        elif xtype == 'audio': # 如果目标是音频
            x = net.audioldm_decode(z) # 使用网络的 audioldm_decode 方法从潜变量解码音频（可能是梅尔频谱图）
            x = net.mel_spectrogram_to_waveform(x) # 将解码得到的梅尔频谱图转换为音频波形
            return x # 返回音频波形 (NumPy 数组)

    def inference(self, xtype=[], condition=[], condition_types=[], n_samples=1, mix_weight={'video': 1, 'audio': 1, 'text': 1, 'image': 1}, image_size=256, ddim_steps=50, scale=7.5, num_frames=8):
        # 定义推理（生成）方法
        # xtype: 一个列表，包含希望生成的目标模态类型，例如 ['image', 'audio']
        # condition: 一个列表，包含输入的条件数据，例如 [PIL.Image 对象, "一段文本", audio_waveform]
        # condition_types: 一个列表，对应 condition 中每个条件的模态类型，例如 ['image', 'text', 'audio']
        # n_samples: 每个目标模态要生成的样本数量
        # mix_weight: (未使用在此片段的采样中，但可能用于更复杂的条件混合) 不同模态条件的混合权重
        # image_size: 生成图像的目标尺寸
        # ddim_steps: DDIM 采样的步数
        # scale: 无条件指导 (classifier-free guidance) 的尺度因子，控制条件对生成结果的影响程度
        # num_frames: 生成视频时的帧数

        net = self.net # 获取 CoDi 网络模型
        sampler = self.sampler # 获取 DDIM 采样器
        ddim_eta = 0.0 # DDIM 采样参数 eta，0.0 表示 DDIM (确定性)，1.0 表示 DDPM (随机性)

        conditioning = [] # 初始化条件嵌入列表，用于存储所有输入条件的编码结果
        # 断言：确保条件类型中没有重复的模态，当前实现可能不支持同一模态的多个条件
        assert len(set(condition_types)) == len(condition_types), "we don't support condition with same modalities yet."
        # 断言：确保条件数据和条件类型列表长度一致
        assert len(condition) == len(condition_types)
        
        # --- 对所有输入条件进行编码 ---
        for i, condition_type in enumerate(condition_types): # 遍历输入的每个条件及其类型
            if condition_type == 'image': # 如果条件是图像
                ctemp1 = regularize_image(condition[i]).cuda() # 对图像进行预处理 (如缩放、归一化) 并移到 GPU
                ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1) # 增加批次维度并复制 n_samples 次
                cim = net.clip_encode_vision(ctemp1).cuda() # 使用 CLIP 编码图像条件
                uim = None # 初始化无条件嵌入
                if scale != 1.0: # 如果使用 classifier-free guidance (scale > 1.0)
                    dummy = torch.zeros_like(ctemp1).cuda() # 创建一个与图像条件形状相同的零张量作为无条件输入
                    uim = net.clip_encode_vision(dummy).cuda() # 编码零张量得到无条件嵌入
                conditioning.append(torch.cat([uim, cim])) # 将无条件嵌入和条件嵌入拼接后加入列表
                                                          # (用于 classifier-free guidance)
                
            elif condition_type == 'video': # 如果条件是视频
                ctemp1 = regularize_video(condition[i]).cuda() # 对视频进行预处理并移到 GPU
                ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1, 1) # 增加批次维度并复制
                cim = net.clip_encode_vision(ctemp1).cuda() # 使用 CLIP (可能是处理视频帧序列的变体) 编码视频条件
                uim = None
                if scale != 1.0:
                    dummy = torch.zeros_like(ctemp1).cuda()
                    uim = net.clip_encode_vision(dummy).cuda()
                conditioning.append(torch.cat([uim, cim]))
                
            elif condition_type == 'audio': # 如果条件是音频
                ctemp = condition[i][None].repeat(n_samples, 1, 1) # 增加批次维度并复制 (假设输入音频形状 [T, F] 或 [L])
                cad = net.clap_encode_audio(ctemp) # 使用 CLAP 编码音频条件
                uad = None
                if scale != 1.0:
                    dummy = torch.zeros_like(ctemp)
                    uad = net.clap_encode_audio(dummy)  
                conditioning.append(torch.cat([uad, cad]))
                
            elif condition_type == 'text': # 如果条件是文本
                # 将文本字符串复制 n_samples 次，然后使用 CLIP 编码
                ctx = net.clip_encode_text(n_samples * [condition[i]]).cuda()
                utx = None
                if scale != 1.0:
                    # 使用空字符串作为无条件文本输入
                    utx = net.clip_encode_text(n_samples * [""]).cuda()
                conditioning.append(torch.cat([utx, ctx]))
        
        # --- 准备目标输出模态的潜变量形状 ---
        shapes = [] # 初始化形状列表
        for xtype_i in xtype: # 遍历希望生成的目标模态
            if xtype_i == 'image': # 如果目标是图像
                h, w = [image_size, image_size] # 获取图像高宽
                shape = [n_samples, 4, h//8, w//8] # 定义图像潜变量的形状
                                                   # 4 通常是 AutoKL 的潜空间通道数
                                                   # h//8, w//8 是因为 AutoKL 通常有 8 倍的下采样
            elif xtype_i == 'video': # 如果目标是视频
                h, w = [image_size, image_size]
                shape = [n_samples, 4, num_frames, h//8, w//8] # 定义视频潜变量的形状 (增加帧维度)
            elif xtype_i == 'text': # 如果目标是文本
                n = 768 # 文本潜变量的维度 (例如 Optimus 或 CLIP 文本特征维度)
                shape = [n_samples, n]
            elif xtype_i == 'audio': # 如果目标是音频
                h, w = [256, 16] # 音频潜变量（频谱图）的典型高宽 (频率 x 时间块)
                                 # 这些值取决于 AudioLDM 的具体配置
                shape = [n_samples, 8, h, w] # 定义音频潜变量的形状 (8 是潜空间通道数)
            else:
                raise # 如果是不支持的目标类型，则抛出异常
            shapes.append(shape) # 将形状加入列表
        

        # --- 使用 DDIM 采样器从噪声生成潜变量 z ---
        z, _ = sampler.sample( # 调用采样器的 sample 方法
            steps=ddim_steps, # DDIM 步数
            shape=shapes, # 目标潜变量形状列表 (对应每个 xtype_i)
            condition=conditioning, # 编码后的条件嵌入列表
            unconditional_guidance_scale=scale, # classifier-free guidance 尺度
            xtype=xtype, # 目标输出模态类型列表
            condition_types=condition_types, # 输入条件模态类型列表
            eta=ddim_eta, # DDIM eta 参数
            verbose=False, # 是否打印详细采样过程
            mix_weight=mix_weight # (可能未使用或用于更复杂的采样策略)
        )
        # z 返回的是一个潜变量列表，每个元素对应一个 xtype_i

        out_all = [] # 初始化所有输出结果的列表
        for i, xtype_i in enumerate(xtype): # 遍历每个目标模态及其生成的潜变量
            z[i] = z[i].cuda() # 将当前模态的潜变量移到 GPU
            x_i = self.decode(z[i], xtype_i) # 调用 decode 方法从潜变量解码得到具体数据
            out_all.append(x_i) # 将解码后的结果加入列表
        return out_all # 返回所有生成结果的列表