#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
#   Editor      : PyCharm
#   File name   : ious.py
#   Author      : Koap
#   Created date: 2020/8/19 下午3:00
#   Description :
#
# ================================================================

import tensorflow as tf
import numpy as np


def bbox_iou(boxes1, boxes2, method='iou'):
    '''
    :param boxes1:  box: x(center)y(center)wh
    :param boxes2:  box: x(center)y(center)wh
    :param method: optional:'iou'/'giou'/'diou'/'ciou'
    :return:  ndims-1
    '''

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes_xyxy_1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                              boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes_xyxy_2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                              boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes_xyxy_1[..., :2], boxes_xyxy_2[..., :2])
    right_down = tf.minimum(boxes_xyxy_1[..., 2:], boxes_xyxy_2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-10)
    iou = tf.clip_by_value(iou,0.,1.)
    if method == 'iou':
        return iou
    enclose_left_up = tf.minimum(boxes_xyxy_1[..., :2], boxes_xyxy_2[..., :2])
    enclose_right_down = tf.maximum(boxes_xyxy_1[..., 2:], boxes_xyxy_2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-10)
    giou = tf.clip_by_value(giou, -1., 1.)
    if method == 'giou':
        return giou
    distance_enclose = tf.reduce_sum(enclose ** 2, axis=-1)
    distance_center = tf.reduce_sum((boxes1[..., :2] - boxes2[..., :2]) ** 2, axis=-1)
    diou = iou - distance_center / (distance_enclose + 1e-10)
    diou = tf.clip_by_value(diou, -1., 1.)
    if method == 'diou':
        return diou
    v = 4 / (np.pi ** 2) * tf.square(tf.subtract(tf.atan(boxes1[..., 2] / boxes1[..., 3]),
                                                 tf.atan(boxes2[..., 2] / boxes2[..., 3])))
    ciou = diou - v ** 2 / (1 - iou + v)
    ciou = tf.clip_by_value(ciou, -1., 1.)
    if method == 'ciou':
        return ciou


def bbox_iou_np(boxes1, boxes2, method='iou'):
    '''
    Numpy implement of iou
    :param boxes1: box: x(center)y(center)wh
    :param boxes2: box: x(center)y(center)wh
    :param method: optional:'iou'/'giou'/'diou'/'ciou'
    :return:
    '''
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes_xyxy_1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                   boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes_xyxy_2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                   boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes_xyxy_1[..., :2], boxes_xyxy_2[..., :2])
    right_down = np.minimum(boxes_xyxy_1[..., 2:], boxes_xyxy_2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-10)
    iou = np.clip(iou, 0., 1.)
    if method == 'iou':
        return iou
    enclose_left_up = np.minimum(boxes_xyxy_1[..., :2], boxes_xyxy_2[..., :2])
    enclose_right_down = np.maximum(boxes_xyxy_1[..., 2:], boxes_xyxy_2[..., 2:])
    enclose = np.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-10)
    giou = np.clip(giou, -1., 1.)
    if method == 'giou':
        return giou
    distance_enclose = np.sum(enclose ** 2, axis=-1)
    distance_center = np.sum((boxes1[..., :2] - boxes2[..., :2]) ** 2, axis=-1)
    diou = iou - distance_center / (distance_enclose + 1e-10)
    diou = np.clip(diou, -1., 1.)
    if method == 'diou':
        return diou
    v = 4 / (np.pi ** 2) * np.square(np.subtract(np.arctan(boxes1[..., 2] / boxes1[..., 3]),
                                                 np.arctan(boxes2[..., 2] / boxes2[..., 3])))
    ciou = diou - v ** 2 / (1 - iou + v)
    ciou = np.clip(ciou, -1., 1.)
    if method == 'ciou':
        return ciou
