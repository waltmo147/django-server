# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from xml.etree.cElementTree import XML, fromstring, tostring, ElementTree
import urllib.request
import urllib
from treelib import Node, Tree
import math
import os
import shutil
import time
import gc

import argparse
import sys

import numpy as np
import tensorflow as tf

def reCalculate_avr(node, treeIn):
    if node.is_leaf():
        node.data.Confidence['Hierarchy'] = node.data.Confidence['Flat']
        return node.data.Confidence['Hierarchy']
    children = treeIn.children(node.identifier)
    sum = 0
    for child in children:
        sum = sum + reCalculate_avr(child, treeIn)
    sum = sum + node.data.Confidence['Flat']
    length = len(children)
    node.data.Confidence['Hierarchy'] = sum / length
    return node.data.Confidence['Hierarchy']


def reCalculate_avr1(node, treeIn):
    if node.is_leaf():
        node.data.Confidence['Hierarchy'] = node.data.Confidence['Flat']
        return node.data.Confidence['Hierarchy']
    children = treeIn.children(node.identifier)
    sum = 0
    datas = []


    for child in children:
        datas.append(reCalculate_avr(child, treeIn))
    datas.sort(reverse=True)
    l = min(5, len(children))

    for i in range(l):
    	sum = sum + datas[i]
    
    sum = sum + node.data.Confidence['Flat']
   
    node.data.Confidence['Hierarchy'] = sum / l
    return node.data.Confidence['Hierarchy']


def reCalculate_sum1(node, treeIn):
    if node.is_leaf():
        node.data.Confidence['Hierarchy_sum'] = node.data.Confidence['Flat']
        return node.data.Confidence['Hierarchy_sum']
    children = treeIn.children(node.identifier)
    sum = 0
    for child in children:
        sum = sum + reCalculate_sum(child, treeIn)
    sum = sum + node.data.Confidence['Flat']
    node.data.Confidence['Hierarchy_sum'] = sum
    return node.data.Confidence['Hierarchy_sum']

def reCalculate_sum(node, treeIn):
    if node.is_leaf():
        node.data.Confidence['Hierarchy_sum'] = node.data.Confidence['Flat']
        return node.data.Confidence['Hierarchy_sum']
    children = treeIn.children(node.identifier)
    sum = 0

    datas = []
    for child in children:
        datas.append(reCalculate_sum(child, treeIn))
    datas.sort(reverse=True)
    l = min(5, len(children))

    for i in range(l):
    	sum = sum + datas[i]

    sum = sum + node.data.Confidence['Flat']

    node.data.Confidence['Hierarchy_sum'] = sum
    return node.data.Confidence['Hierarchy_sum']

def load_graph(model_file):
  graph = tf.Graph()

  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)


  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)

  
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')

  float_caster = tf.cast(image_reader, tf.float32)

  dims_expander = tf.expand_dims(float_caster, 0);

  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])


  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])


  sess = tf.Session()

  # with tf.Session() as sess:
  #   sess.run(...)

  result = sess.run(normalized)
  #sess.close()
  

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label



# building tree
class Confidence(object):
  def __init__(self, a = 0, b = 0, c = 0):
    self.Confidence = {'Flat': a, 'Hierarchy': b, 'Hierarchy_sum': c}

# with urllib.request.urlopen('http://www.image-net.org/api/xml/structure_released.xml') as response:
#     html = response.read()
f11 = open('treetext.txt','rb')
html = f11.read()
f11.close()
tree = ElementTree(fromstring(html))
root = tree.getroot()

synsetTree = Tree()

synsetTree.create_node('Entity', 'fall11', data = Confidence(0, 0))
for synset in root.iter('synset'):
  for child in synset:
    if child.get('wnid') in synsetTree._nodes:
      continue
    synsetTree.create_node(child.get('words'), child.get('wnid'), parent = synset.get('wnid'), data = Confidence(0, 0))

# synsetTree.show()
# treeDog = synsetTree.subtree('n02087122')
treeDog = synsetTree
model_file = "tf_files/retrained_graph.pb"
graph = load_graph(model_file)
#graph.finalize()


def image_label(file_ID, file_suffix, graph):
  if __name__ == "__main__":
    start = time.clock()
    file_name = "tf_files/ImageNet200_test" + '/' + file_ID + '/' + file_suffix
    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    result_with_tree_avr = 0
    result_without_tree = 0
    result_with_tree_sum = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    if args.graph:
      model_file = args.graph
    if args.image:
      file_name = args.image
    if args.labels:
      label_file = args.labels
    if args.input_height:
      input_height = args.input_height
    if args.input_width:
      input_width = args.input_width
    if args.input_mean:
      input_mean = args.input_mean
    if args.input_std:
      input_std = args.input_std
    if args.input_layer:
      input_layer = args.input_layer
    if args.output_layer:
      output_layer = args.output_layer

    

    t = read_tensor_from_image_file(file_name,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer

    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);




    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
    results = np.squeeze(results)

    top_k = results.argmax()
    labels = load_labels(label_file)
    # for i in top_k:
    #   print(labels[i], results[i])

    if labels[top_k] == file_ID:
      result_without_tree = 1



    len_results = len(results)
    for i in range(len_results):
      treeDog.get_node(labels[i]).data.Confidence['Flat'] = results[i]

    treeDog.get_node(treeDog.root).data.Confidence['Hierarchy'] = reCalculate_avr(treeDog.get_node(treeDog.root), treeDog)
    treeDog.get_node(treeDog.root).data.Confidence['Hierarchy_sum'] = reCalculate_sum(treeDog.get_node(treeDog.root), treeDog)

    pointer = treeDog.get_node(treeDog.root)
    if pointer.identifier == file_ID:
      result_with_tree_avr = 1;
    #print (pointer.tag)
    while (not pointer.is_leaf()):
      children = treeDog.children(pointer.identifier)
      List = []
      for i in range(len(children)):
        List.append(children[i].data.Confidence['Hierarchy'])
      pointer = children[List.index(max(List))]
      if pointer.identifier == file_ID:
      	result_with_tree_avr = 1;



    pointer = treeDog.get_node(treeDog.root)
    if pointer.identifier == file_ID:
      result_with_tree_sum = 1;

    while (not pointer.is_leaf()):
      children = treeDog.children(pointer.identifier)
      List = []
      for i in range(len(children)):
        List.append(children[i].data.Confidence['Hierarchy_sum'])
      pointer = children[List.index(max(List))]
      if pointer.identifier == file_ID:
      	result_with_tree_sum = 1;
    #  print (pointer.tag)


    return [result_without_tree, result_with_tree_sum, result_with_tree_avr]


firststart = time.clock()
print ('aaaaaaa')
path_img='tf_files/ImageNet200_test'
ls = os.listdir(path_img)
lenl = len(ls)
total = 0
with_tree = 0
with_tree_sum = 0
without_tree = 0
list_sum = []
list_avr = []

file_res = open('testfile.txt','w') 
 


for i in range(lenl):
    if ls[i] == '.DS_Store':
        continue

    if not treeDog.get_node(ls[i]).is_leaf():
        continue

    ls_sub = os.listdir(path_img + '/' + ls[i])
    lenl_sub = (int)(len(ls_sub))
    temp_without_tree = 0
    temp_with_tree = 0
    temp_with_tree_sum = 0
    temp_total = 0
    for j in range(lenl_sub):
        if ls_sub[j] == '.DS_Store':
            continue
        
        result = image_label(ls[i], ls_sub[j], graph)
        tf.reset_default_graph()

        print (result)
        temp_total += 1
        temp_without_tree += result[0]
        temp_with_tree_sum += result[1]
        temp_with_tree += result[2]

    total += temp_total
    without_tree += temp_without_tree
    with_tree += temp_with_tree
    with_tree_sum += temp_with_tree_sum

    if temp_with_tree_sum > temp_without_tree:
        list_sum.append(ls[i])
    if temp_with_tree > temp_without_tree:
        list_avr.append(ls[i])

    

    if temp_total != 0:
        temp_accuracy = [temp_without_tree / temp_total, temp_with_tree_sum / temp_total, temp_with_tree / temp_total]
    print ('accuracy for %s :' % ls[i], temp_accuracy)
    if (temp_accuracy[0] < temp_accuracy[1]):
      file_res.write('sum:' + ls[i]) 
    if (temp_accuracy[0] < temp_accuracy[2]):
      file_res.write('avr:' + ls[i]) 
    print ('count: %s' % total)
    accuracy = [without_tree / total, with_tree_sum / total, with_tree / total]
    print ('total: %s' % accuracy)


#gc.enable()
print ('sum')
print (list_sum)
print ('average')
print (list_avr)

finalend = time.clock()
print (finalend - firststart)



file_res.close() 



