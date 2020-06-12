import SimpleITK as sitk

itk_img = sitk.ReadImage('data/SHFK_298474/SHFK_298474.nii')
img = sitk.GetArrayFromImage(itk_img)
print(img.shape)
origin = itk_img.GetOrigin()
spacing = itk_img.GetSpacing()
x = origin[0]
y = origin[1]
z = origin[2]
x_spacing = spacing[0]
y_spacing = spacing[1]
z_spacing = spacing[2]
x_slice_min = int((abs(x) + (abs(-96.6166) - 12.9472))/x_spacing)
x_slice_max = int((abs(x) + (abs(-96.6166) + 12.9472))/x_spacing)
y_slice_min = int((abs(y) - (abs(114.023) + 15.0021))/y_spacing)
y_slice_max = int((abs(y) - (abs(114.023) - 15.0021))/y_spacing)
z_slice_min = int((abs(z)-(abs(-114.494) + 12.8726))/z_spacing)
z_slice_max = int((abs(z)-(abs(-114.494) - 12.8726))/z_spacing)
print(x)
print(y)
print(z)
print(x_slice_min,x_slice_max)
print(y_slice_min,y_slice_max)
print(z_slice_min,z_slice_max)

roi = img[z_slice_min:z_slice_max,x_slice_min:x_slice_max,y_slice_min:y_slice_max]
print('roi shape',roi.shape)

sitk.WriteImage(itk_img[x_slice_min:x_slice_max,y_slice_min:y_slice_max,z_slice_min:z_slice_max],'1.mhd')
# plt.imshow(itk_img[:, :,358], cmap='gray')
# plt.show()

