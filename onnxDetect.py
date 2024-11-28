import cv2
import onnxruntime as ort
import numpy as np
import argparse 
# 置信度
confidence_thres = 0.35
# iou阈值
iou_thres = 0.5
# 类别
classes = {"car","truck","bus","van","freight"}
# 随机颜色
color_palette = np.random.uniform(100, 255, size=(len(classes), 3))

# 判断是使用GPU或CPU
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,  # 可以选择GPU设备ID，如果你有多个GPU
    }),
    'CPUExecutionProvider',  # 也可以设置CPU作为备选
]

def calculate_iou(box, other_boxes):
    """
    计算给定边界框与一组其他边界框之间的交并比（IoU）。

    参数：
    - box: 单个边界框，格式为 [x1, y1, width, height]。
    - other_boxes: 其他边界框的数组，每个边界框的格式也为 [x1, y1, width, height]。

    返回值：
    - iou: 一个数组，包含给定边界框与每个其他边界框的IoU值。
    """

    # 计算交集的左上角坐标
    x1 = np.maximum(box[0], np.array(other_boxes)[:, 0])
    y1 = np.maximum(box[1], np.array(other_boxes)[:, 1])
    # 计算交集的右下角坐标
    x2 = np.minimum(box[0] + box[2], np.array(other_boxes)[:, 0] + np.array(other_boxes)[:, 2])
    y2 = np.minimum(box[1] + box[3], np.array(other_boxes)[:, 1] + np.array(other_boxes)[:, 3])
    # 计算交集区域的面积
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    # 计算给定边界框的面积
    box_area = box[2] * box[3]
    # 计算其他边界框的面积
    other_boxes_area = np.array(other_boxes)[:, 2] * np.array(other_boxes)[:, 3]
    # 计算IoU值
    iou = intersection_area / (box_area + other_boxes_area - intersection_area)
    return iou

def custom_NMSBoxes(boxes, scores, confidence_threshold, iou_threshold):
    # 如果没有边界框，则直接返回空列表
    if len(boxes) == 0:
        return []
    # 将得分和边界框转换为NumPy数组
    scores = np.array(scores)
    boxes = np.array(boxes)
    # 根据置信度阈值过滤边界框
    mask = scores > confidence_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    # 如果过滤后没有边界框，则返回空列表
    if len(filtered_boxes) == 0:
        return []
    # 根据置信度得分对边界框进行排序
    sorted_indices = np.argsort(filtered_scores)[::-1]
    # 初始化一个空列表来存储选择的边界框索引
    indices = []
    # 当还有未处理的边界框时，循环继续
    while len(sorted_indices) > 0:
        # 选择得分最高的边界框索引
        current_index = sorted_indices[0]
        indices.append(current_index)
        # 如果只剩一个边界框，则结束循环
        if len(sorted_indices) == 1:
            break
        # 获取当前边界框和其他边界框
        current_box = filtered_boxes[current_index]
        other_boxes = filtered_boxes[sorted_indices[1:]]
        # 计算当前边界框与其他边界框的IoU
        iou = calculate_iou(current_box, other_boxes)
        # 找到IoU低于阈值的边界框，即与当前边界框不重叠的边界框
        non_overlapping_indices = np.where(iou <= iou_threshold)[0]
        # 更新sorted_indices以仅包含不重叠的边界框
        sorted_indices = sorted_indices[non_overlapping_indices + 1]
    # 返回选择的边界框索引
    return indices

# 绘制单张图像
# def draw_detections(img, box, score, class_id):
#     """
#     在输入图像上绘制检测到的对象的边界框和标签。

#     参数:
#             img: 要在其上绘制检测结果的输入图像。
#             box: 检测到的边界框。
#             score: 对应的检测得分。
#             class_id: 检测到的对象的类别ID。

#     返回:
#             无
#     """

#     # 提取边界框的坐标
#     x1, y1, w, h = box
#     # 根据类别ID检索颜色
#     color = color_palette[class_id]
#     # 在图像上绘制边界框
#     cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
#     # 创建标签文本，包括类名和得分
#     label = f'{classes[class_id]}: {score:.2f}'
#     # 计算标签文本的尺寸
#     (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#     # 计算标签文本的位置
#     label_x = x1
#     label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
#     # 绘制填充的矩形作为标签文本的背景
#     cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
#     # 在图像上绘制标签文本
#     cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

color_palette = [  
            [165, 0, 255],       
            [0, 255, 0],        
            [102, 255, 255],       
            [255, 165, 0],      
            [255, 255, 0]      
        ]# 示例颜色，可以根据需要扩展
classes = ["car","truck","bus","van","freight"]   # 示例类别名称，可以根据需要扩展
 
def draw_detections_on_combined_image(combined_img,offset, box, score, class_id):
    """
    将两个输入图像水平拼接，并在合并后的图像上绘制检测到的对象的边界框和标签。
 
    参数:
            img1, img2: 要合并并绘制检测结果的输入图像。
            box: 检测到的边界框，形如 (x, y, w, h)。
            score: 检测得分。
            class_id: 检测到的对象的类别ID。
 
    返回:
            combined_img: 合并后并绘制了检测结果的图像。
    """
    
 
    # 绘制检测结果
    x, y, w, h = box
    color = color_palette[class_id]
    label = f'{classes[class_id]}: {score:.2f}'
 
    # 调整边界框坐标以适应合并后的图像（如果需要）
    # 这里假设边界框是针对img1的，如果它是针对img2的，则需要加上w1的偏移
    # 针对img1（或可以添加逻辑来区分）
    adjusted_x = x 
    adjusted_y = y
    adjusted_w = w 
    adjusted_h = h
 
    cv2.rectangle(combined_img, (int(adjusted_x), int(adjusted_y)), (int(adjusted_x + adjusted_w), int(adjusted_y + adjusted_h)), color, 2)
 
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = adjusted_x
    label_y = adjusted_y - 10 if adjusted_y - 10 > label_height else adjusted_y + 10
    cv2.rectangle(combined_img, (label_x, label_y - label_height), (label_x + label_width, label_y), color, cv2.FILLED)
    cv2.putText(combined_img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # 如果box是针对img2，则应为 x + w1
    adjusted_x = x +offset 
    adjusted_y = y
    adjusted_w = w
    adjusted_h = h

    cv2.rectangle(combined_img, (int(adjusted_x), int(adjusted_y)), (int(adjusted_x + adjusted_w), int(adjusted_y + adjusted_h)), color, 2)
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = adjusted_x
    label_y = adjusted_y - 10 if adjusted_y - 10 > label_height else adjusted_y + 10
    cv2.rectangle(combined_img, (label_x, label_y - label_height), (label_x + label_width, label_y), color, cv2.FILLED)
    cv2.putText(combined_img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return combined_img

def preprocess(img, input_width, input_height):
    """
    在执行推理之前预处理输入图像。

    返回:
        image_data: 为推理准备好的预处理后的图像数据。
    """

    # 获取输入图像的高度和宽度
    img_height, img_width = img.shape[:2]

    # 将图像颜色空间从BGR转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将图像大小调整为匹配输入形状
    img = cv2.resize(img, (input_width, input_height))
    # 通过除以255.0来归一化图像数据
    image_data = np.array(img) / 255.0
    # 转置图像，使通道维度为第一维
    image_data = np.transpose(image_data, (2, 0, 1))  # 通道首
    # 扩展图像数据的维度以匹配预期的输入形状
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    # 返回预处理后的图像数据
    return image_data, img_height, img_width

def postprocess(rgb_input_image,ir_input_image, output, input_width, input_height, img_width, img_height):
    """
    对模型输出进行后处理，提取边界框、得分和类别ID。

    参数:
        input_image (numpy.ndarray): 输入图像。
        output (numpy.ndarray): 模型的输出。
        input_width (int): 模型输入宽度。
        input_height (int): 模型输入高度。
        img_width (int): 原始图像宽度。
        img_height (int): 原始图像高度。

    返回:
        numpy.ndarray: 绘制了检测结果的输入图像。
    """

    # 转置和压缩输出以匹配预期的形状
    outputs = np.transpose(np.squeeze(output[0]))
    # 获取输出数组的行数
    rows = outputs.shape[0]
    # 用于存储检测的边界框、得分和类别ID的列表
    boxes = []
    scores = []
    class_ids = []
    # 计算边界框坐标的缩放因子
    x_factor = img_width / input_width
    y_factor = img_height / input_height

    # 确定拼接后的图像大小
    h1, w1 = rgb_input_image.shape[:2]
    h2, w2 = ir_input_image.shape[:2]
    combined_height = max(h1, h2)
    combined_width = w1 + w2
 
    # 创建一个空白的合并图像
    combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
 
    # 将两个图像复制到合并图像的相应位置
    combined_img[:h1, :w1] = rgb_input_image
    combined_img[:h2, w1:w1+w2] = ir_input_image if h2 <= h1 else ir_input_image[:h1, :]

    # 遍历输出数组的每一行
    for i in range(rows):
        # 从当前行提取类别得分
        classes_scores = outputs[i][4:]
        # 找到类别得分中的最大得分
        max_score = np.amax(classes_scores)
        # 如果最大得分高于置信度阈值
        if max_score >= confidence_thres:
            # 获取得分最高的类别ID
            class_id = np.argmax(classes_scores)
            # 从当前行提取边界框坐标
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            # 计算边界框的缩放坐标
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            # 将类别ID、得分和框坐标添加到各自的列表中
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])
    # 应用非最大抑制过滤重叠的边界框
    indices = custom_NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    # 遍历非最大抑制后的选定索引
    for i in indices:
        # 根据索引获取框、得分和类别ID
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        # 在输入图像上绘制检测结果
        combined_img=draw_detections_on_combined_image(combined_img,w1,box, score, class_id)
 
    # 返回修改后的输入图像
    return combined_img

def init_detect_model(model_path):
    # 使用ONNX模型文件创建一个推理会话，并指定执行提供者
    session = ort.InferenceSession(model_path, providers=providers)
    # 获取模型的输入信息
    model_inputs = session.get_inputs()
    # 获取输入的形状，用于后续使用
    input_shape = model_inputs[0].shape
    # 从输入形状中提取输入宽度
    input_width = input_shape[2]
    # 从输入形状中提取输入高度
    input_height = input_shape[3]
    # 返回会话、模型输入信息、输入宽度和输入高度
    return session, model_inputs, input_width, input_height

def detect_object(rgb_image,ir_image, session, model_inputs, input_width, input_height):
    rgb_result_image = rgb_image
    ir_result_image = ir_image

    # 预处理图像图像数据，调整图像大小并可能进行归一化等操作
    # 将可见光和红外图像执行相同的前处理
    rgb_data, img_height, img_width = preprocess(rgb_image, input_width, input_height)
    ir_data, img_height, img_width = preprocess(ir_image, input_width, input_height)
    # 拼接可见光和红外光
    img_data= np.concatenate((rgb_data, ir_data), axis=1)
    # 使用预处理后的图像数据进行推理
    outputs = session.run(None, {model_inputs[0].name: img_data})
    # 对推理结果进行后处理，例如解码检测框，过滤低置信度的检测等
    output_image = postprocess(rgb_result_image,ir_result_image, outputs, input_width, input_height, img_width, img_height)
    # 返回处理后的图像
    return output_image
if __name__ == '__main__':
    # 模型文件的路径
    parser = argparse.ArgumentParser(description='TwoStream detect')
    parser.add_argument('--onnx', default="", type=str, help='onnx file name')
    parser.add_argument('--rgbfile', default="", type=str, help='rgb file name')
    parser.add_argument('--irfile', default="", type=str, help='rgb file name')
    args = parser.parse_args()

    model_path = args.onnx

    # 初始化检测模型，加载模型并获取模型输入节点信息和输入图像的宽度、高度
    session, model_inputs, input_width, input_height = init_detect_model(model_path)
    # 读取可见光图像文件

    rgb_data = cv2.imread(args.rgbfile)
    # 读取红外光图像文件
    ir_data = cv2.imread(args.irfile)
   
    # 使用检测模型对读入的图像进行对象检测
    result_image = detect_object(rgb_data,ir_data, session, model_inputs, input_width, input_height)
    # 将检测后的图像保存到文件
    cv2.imwrite("output_image.jpg", result_image)
    # 等待用户按键，然后关闭显示窗口
    cv2.waitKey(0)
    



