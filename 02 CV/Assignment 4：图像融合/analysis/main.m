%确定输入的可见光图像路径、红外图像路径、融合图像路径
path ='../images/21_pairs_tno/';
ir_path = strcat(path,'ir/');
vis_path = strcat(path,'vis/');
fus_path = '../outputs/alpha_1e4_21/fused_rfnnest_700_wir_6.0_wvi_3.0_21_res/fused_rfnnest_700_wir_6.0_wvi_3.0_';
source = dir(ir_path);
source1_imgs = string(zeros(1,length(source)-2));
source2_imgs = string(zeros(1,length(source)-2));
source3_imgs = string(zeros(1,length(source)-2));
%确定对应的各个图像的路径
for i=3:length(source)
    source1_imgs(i-2) = source(i).name;
    source2_imgs(i-2) = strcat(vis_path,strrep(source1_imgs(i-2),'IR','VIS'));
    source3_imgs(i-2) = strcat(fus_path,source1_imgs(i-2));
    source1_imgs(i-2) = strcat(ir_path,source1_imgs(i-2));
end

%结果保存文件
f = fopen("../outputs/alpha_1e4_21/analysis.txt","w");

%分别对各个图像融合情况计算
result = [0,0,0,0,0,0]
for i=1:length(source1_imgs)
    fprintf("start compute the image%d\n",i);
    %读取图像
    fileName_source_l = source1_imgs(i);
    fileName_source_r = source2_imgs(i);
    fileName_fused = source3_imgs(i);
    fused_image = imread(convertStringsToChars(fileName_fused));
    sourceTestImage1 = imread(convertStringsToChars(fileName_source_l));
    sourceTestImage2 = imread(convertStringsToChars(fileName_source_r));
    %对图像进行分析
    metrics = analysis_Reference(fused_image,sourceTestImage1,sourceTestImage2);
    %保存结果
    result = result + [metrics.EN, metrics.SD, metrics.MI, metrics.Nabf, metrics.SCD, metrics.MS_SSIM];
    fprintf("The analysis is\t EN:%f SD:%f MI:%f Nabf:%f SCD:%f MS_SSIM:%f\n",metrics.EN, metrics.SD, metrics.MI, metrics.Nabf, metrics.SCD, metrics.MS_SSIM)
    fprintf(f,"The analysis is\t EN:%f SD:%f MI:%f Nabf:%f SCD:%f MS_SSIM:%f\n",metrics.EN, metrics.SD, metrics.MI, metrics.Nabf, metrics.SCD, metrics.MS_SSIM);
end

%计算平均值并输出
result = result/length(source1_imgs)
fprintf("The average analysis is\t %f %f %f %f %f %f\n",result(1),result(2),result(3),result(4),result(5),result(6))
fprintf(f,"The average analysis is\t %f %f %f %f %f %f\n",result(1),result(2),result(3),result(4),result(5),result(6));
fclose(f);