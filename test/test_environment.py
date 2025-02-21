if __name__ == '__main__':
    import torch

    print(torch.__version__)  # 查看 PyTorch 版本
    print(torch.cuda.is_available())  # 输出应为 True 才支持 CUDA