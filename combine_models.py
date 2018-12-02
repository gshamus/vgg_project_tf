import tensorflow as tf
from tensorflow.python.framework import meta_graph as mg

vgg_file_path ='../tf_model_info/tf_out/'
vgg_in_tensor_name = 'input:0'
vgg_out_tensor_name = 'Flatten_2/flatten/Reshape:0'

gc_file_path = './model_1e-05_20_0.25/best_model/'
gc_in_tensor_name = 'inputs/Placeholder:0'
gc_out_tensor_name = 'output/Softmax:0'

write_dir = "./end_to_end/"
graph = tf.get_default_graph()
with tf.Session() as sess:
    full_graph_input = tf.placeholder(tf.float32, (None, 224, 224, 3), name = 'full_graph_input_ph')
    vgg_meta = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], vgg_file_path, import_scope = 'vgg_org')
    gc_meta = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], gc_file_path, import_scope = 'gc_org')

    mg.import_scoped_meta_graph(vgg_meta, input_map = {vgg_in_tensor_name : full_graph_input}, import_scope = 'vgg_graph')
    vgg_out = graph.get_tensor_by_name('vgg_graph/' + vgg_out_tensor_name)

    mg.import_scoped_meta_graph(gc_meta, input_map = {gc_in_tensor_name : vgg_out}, import_scope = 'gc_graph')
    final_out  = graph.get_tensor_by_name('gc_graph/' + gc_out_tensor_name)

    #sess.run(tf.initialize_variables([full_graph_input]))
    tf.saved_model.simple_save(sess, write_dir,
                                    inputs = {'full_input' : full_graph_input},
                                    outputs = {'softmax_probs' : final_out}
                                    )


"""
graph = tf.get_default_graph()
full_graph_input = tf.placeholder(tf.float32, (None, 224, 224, 3), name = 'full_graph_input_ph')

with tf.Session(graph=tf.Graph()) as sess1:
    vgg_meta = tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.TRAINING], vgg_file_path)


with tf.Session(graph=tf.Graph()) as sess2:
    gc_meta = tf.saved_model.loader.load(sess2, [tf.saved_model.tag_constants.SERVING], gc_file_path)

mg.import_scoped_meta_graph(vgg_meta, input_map = {vgg_in_tensor_name : full_graph_input}, import_scope = 'vgg_graph')
vgg_out = graph.get_tensor_by_name('vgg_graph/' + vgg_out_tensor_name)

mg.import_scoped_meta_graph(gc_meta, input_map = {gc_in_tensor_name : vgg_out}, import_scope = 'gc_graph/')
final_out  = graph.get_tensor_by_name('gc_graph/' + gc_out_tensor_name)

write_dir = "./end_to_end/"
with tf.Session(graph = graph) as sess:
    sess.run(tf.global_variables_initializer())
    tf.saved_model.simple_save(sess, write_dir,
                                    inputs = {'full_input' : full_graph_input},
                                    outputs = {'softmax_probs' : final_out}
                                    )
"""



#tf.reset_default_graph()

"""
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], vgg_file_path)
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], gc_file_path)
    graph_def = sess.graph.as_graph_def()
    print("VGG IN ", sess.graph.get_tensor_by_name(vgg_in_tensor_name))
    print("VGG OUT ", sess.graph.get_tensor_by_name(vgg_out_tensor_name))
    print("GC IN ", sess.graph.get_tensor_by_name(gc_in_tensor_name))
    print("GC OUT ", sess.graph.get_tensor_by_name(gc_out_tensor_name))
    layers = [op.name for op in sess.graph.get_operations()]
    print(layers)
"""
"""
with tf.Session(graph=tf.Graph()) as sess1:
    vgg_meta = tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.TRAINING], vgg_file_path)
    vgg_graph_def = sess1.graph.as_graph_def()

with tf.Session(graph=tf.Graph()) as sess2:
    gc_meta = tf.saved_model.loader.load(sess2, [tf.saved_model.tag_constants.SERVING], gc_file_path)
    gc_graph_def = sess2.graph.as_graph_def()

with tf.Session(graph = tf.Graph()) as sess:



c_graph = tf.Graph()

with c_graph.as_default():
    vgg_saver = tf.train.import_meta_graph(vgg_meta)
    gc_saver =  tf.train.import_meta_graph(gc_meta)
    vgg_in = c_graph.get_tensor_by_name(vgg_in_tensor_name)
    vgg_in_var = tf.Variable(vgg_in, validate_shape = False)

    print(vgg_in)
    vgg_out = c_graph.get_tensor_by_name(vgg_out_tensor_name)
    print(vgg_out)
    vgg_out_var = tf.Variable(vgg_out, validate_shape = False)
    gc_in = c_graph.get_tensor_by_name(gc_in_tensor_name)
    gc_in_var = tf.Variable(gc_in, validate_shape = False)
    gc_out = c_graph.get_tensor_by_name(gc_out_tensor_name)
    gc_out_var =  tf.Variable(gc_out, validate_shape = False)
    

with c_graph.as_default():
    saver_vgg = tf.train.Saver(var_list = [vgg_in_var, vgg_out_var])#tf.train.Saver(var_list = {'vgg_input' : vgg_in, 'vgg_feat_extract' : vgg_out})
    saver_gc = tf.train.Saver(var_list = [gc_in_var, gc_out_var])#tf.train.Saver(var_list = {'gc_input' : gc_in, 'gc_final_out' : gc_out})
    saver_vgg.save()

with tf.session(graph = c_graph) as sess():
    saver_vgg.restore(sess, '.')
"""
"""
vgg_var_path = '../tf_model_info/tf_out/variables/variables.data-00000-of-00001'
with tf.Session(graph=tf.Graph()) as sess1:
    vgg_meta = tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.TRAINING], vgg_file_path)
    vgg_saver = tf.train.import_meta_graph(vgg_meta, clear_devices = True)
    vgg_saver.restore(sess1, vgg_var_path)
    vgg_graph_def = sess1.graph.as_graph_def()
    print("VGG IN ", sess1.graph.get_tensor_by_name(vgg_in_tensor_name))
    print("VGG OUT ", sess1.graph.get_tensor_by_name(vgg_out_tensor_name))

gc_var_path = './model_1e-05_20_0.25/best_model/variables/variables.data-00000-of-00001'
with tf.Session(graph=tf.Graph()) as sess2:
    gc_meta = tf.saved_model.loader.load(sess2, [tf.saved_model.tag_constants.SERVING], gc_file_path)
    gc_saver = tf.train.import_meta_graph(gc_meta, clear_devices = True)
    gc_saver.restore(sess2, gc_var_path)
    gc_graph_def = sess2.graph.as_graph_def()
    print("GC IN ", sess2.graph.get_tensor_by_name(gc_in_tensor_name))
    print("GC OUT ", sess2.graph.get_tensor_by_name(gc_out_tensor_name))

write_dir = "./end_to_end/"
with tf.Session(graph = tf.get_default_graph()) as sess:

    full_graph_input = tf.placeholder(tf.float32, (None, 224, 224, 3))

    [vgg_out] = tf.import_graph_def(vgg_graph_def, 
                        input_map = {vgg_in_tensor_name : full_graph_input},
                        return_elements = [vgg_out_tensor_name]
                        )
    print(vgg_out)

    [final_out] = tf.import_graph_def(gc_graph_def, 
                        input_map = {gc_in_tensor_name : vgg_out},
                        return_elements = [gc_out_tensor_name]
                        )
    print(final_out)

    tf.saved_model.simple_save(sess, write_dir,
                                    inputs = {'input' : full_graph_input},
                                    outputs = {'softmax_probs' : final_out}
                                    )

"""
"""
write_dir = "./end_to_end/"
saver = tf.train.Saver()

with tf.Session(graph = tf.get_default_graph()) as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver.save(sess, './combined_model.ckpt')

    tf.saved_model.simple_save(sess, write_dir,
                                    inputs = {'input' : full_graph_input},
                                    outputs = {'softmax_probs' : final_out}
                                    )
"""