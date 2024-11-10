class ToRGBTransform:
    def __call__(self, img):
        # 检查图像的通道数
        if img.mode != 'RGB':  # 灰度图像
            # 转换为三通道图像，比如一通道的灰度图像或者四通道的 rgba 图像
            img = img.convert('RGB')
        return img
