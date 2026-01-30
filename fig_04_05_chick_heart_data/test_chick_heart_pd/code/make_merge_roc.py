from PIL import Image

# 打开两张图片
img1 = Image.open("../output/figures/fig5_roc_sub_a.png")
img2 = Image.open("../output/figures/fig5_roc_sub_b.png")

# 计算新图片的尺寸（按列合并：宽度相加，高度取最大值）
new_width = img1.width + img2.width
new_height = max(img1.height, img2.height)

# 创建新画布
merged = Image.new("RGB", (new_width, new_height))

# 粘贴图片（按列排列：img1在左，img2在右）
merged.paste(img1, (0, 0))  # img1 粘贴到左侧 (x=0, y=0)
merged.paste(img2, (img1.width, 0))  # img2 粘贴到右侧 (x=img1.width, y=0)

# 保存结果
merged.save("../output/figures/fig5_roc.png")
print('合并完成！')
