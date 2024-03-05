# 可视化工具
import cv2


def show_result(img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(255, 255, 241),
                mask_color=None,
                thickness=1,
                font_scale=0.2,
                font_size=11,
                win_name='',
                fig_size=(15, 10),
                show=False,
                wait_time=0,
                out_file=None):
    img = img.copy()

    # 获取图像放大比例
    h, w = img.shape[:2]
    scale_h = h / fig_size[1]
    scale_w = w / fig_size[0]

    # for bbox, label, score in zip(result['boxes'], result['labels'], result['scores']):
    for bbox, score in zip(result[0][:, :4], result[0][:, 4]):
        label = 'class'
        if score >= score_thr:
            bbox = [int(i) for i in bbox]
            left_top = (bbox[0], bbox[1])
            right_bottom = (bbox[2], bbox[3])
            cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)

            label_text = f'{label}: {score:.2f}'
            # ((label_width, label_height), _) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            #                                                    font_size)

            # 动态调整字体大小
            adjusted_font_scale = font_scale * min(scale_h, scale_w)

            ((label_width, label_height), _) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                               adjusted_font_scale,
                                                               font_size)

            # 保证文本在图片内部
            text_x = max(left_top[0], 0)
            text_y = max(left_top[1] - 5, label_height)

            cv2.rectangle(img, left_top, (left_top[0] + label_width, left_top[1] - label_height - 5), bbox_color, -1)
            cv2.putText(img, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        text_color, thickness, lineType=cv2.LINE_AA)

    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, *fig_size)
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)

    if out_file is not None:
        cv2.imwrite(out_file, img)
