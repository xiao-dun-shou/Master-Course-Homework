%ȷ������Ŀɼ���ͼ��·��������ͼ��·�����ں�ͼ��·��
path ='../images/21_pairs_tno/';
ir_path = strcat(path,'ir/');
vis_path = strcat(path,'vis/');
fus_path = '../outputs/alpha_1e4_21/fused_rfnnest_700_wir_6.0_wvi_3.0_21_res/fused_rfnnest_700_wir_6.0_wvi_3.0_';
source = dir(ir_path);
source1_imgs = string(zeros(1,length(source)-2));
source2_imgs = string(zeros(1,length(source)-2));
source3_imgs = string(zeros(1,length(source)-2));
%ȷ����Ӧ�ĸ���ͼ���·��
for i=3:length(source)
    source1_imgs(i-2) = source(i).name;
    source2_imgs(i-2) = strcat(vis_path,strrep(source1_imgs(i-2),'IR','VIS'));
    source3_imgs(i-2) = strcat(fus_path,source1_imgs(i-2));
    source1_imgs(i-2) = strcat(ir_path,source1_imgs(i-2));
end

%��������ļ�
f = fopen("../outputs/alpha_1e4_21/analysis.txt","w");

%�ֱ�Ը���ͼ���ں��������
result = [0,0,0,0,0,0]
for i=1:length(source1_imgs)
    fprintf("start compute the image%d\n",i);
    %��ȡͼ��
    fileName_source_l = source1_imgs(i);
    fileName_source_r = source2_imgs(i);
    fileName_fused = source3_imgs(i);
    fused_image = imread(convertStringsToChars(fileName_fused));
    sourceTestImage1 = imread(convertStringsToChars(fileName_source_l));
    sourceTestImage2 = imread(convertStringsToChars(fileName_source_r));
    %��ͼ����з���
    metrics = analysis_Reference(fused_image,sourceTestImage1,sourceTestImage2);
    %������
    result = result + [metrics.EN, metrics.SD, metrics.MI, metrics.Nabf, metrics.SCD, metrics.MS_SSIM];
    fprintf("The analysis is\t EN:%f SD:%f MI:%f Nabf:%f SCD:%f MS_SSIM:%f\n",metrics.EN, metrics.SD, metrics.MI, metrics.Nabf, metrics.SCD, metrics.MS_SSIM)
    fprintf(f,"The analysis is\t EN:%f SD:%f MI:%f Nabf:%f SCD:%f MS_SSIM:%f\n",metrics.EN, metrics.SD, metrics.MI, metrics.Nabf, metrics.SCD, metrics.MS_SSIM);
end

%����ƽ��ֵ�����
result = result/length(source1_imgs)
fprintf("The average analysis is\t %f %f %f %f %f %f\n",result(1),result(2),result(3),result(4),result(5),result(6))
fprintf(f,"The average analysis is\t %f %f %f %f %f %f\n",result(1),result(2),result(3),result(4),result(5),result(6));
fclose(f);