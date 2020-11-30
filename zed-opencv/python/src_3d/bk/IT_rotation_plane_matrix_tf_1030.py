import open3d,math as mt,numpy as np
import tensorflow.compat.v1  as tf
tf.disable_eager_execution()
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# バイアス変数
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def main(batch_x,batch_y,bt_size):
  x = tf.placeholder(tf.float32, shape=[None, 46960,3])
  y_ = tf.placeholder(tf.float32, shape=[1, 3])

  W_fc2 = weight_variable([3, 3])
  # b_fc2 = bias_variable([3])
  pointsn = tf.matmul(x, W_fc2)

  def get_reg_linear(X, Y, Z):
      XY = tf.transpose(tf.stack([X, Y],axis=1))
      XY = tf.concat((tf.ones((XY.shape[0], 1,1)), XY),axis=1)
      XY=XY[:,:,0]
      XY1=tf.matmul(XY, XY, transpose_a=True)
      XY1=tf.linalg.inv(XY1)
      XY2=tf.matmul(XY , tf.expand_dims(Z[0,:],axis=1),transpose_a=True)
      # bhat = np.linalg.inv(tf.transpose(XY) @ XY) @ XY.T @ Z
      bhat=tf.matmul(XY1,XY2)
      bhat=tf.reshape(bhat,(-1,3))
      return bhat

  X, Y, Z = pointsn[:,:,0],pointsn[:,:,1],pointsn[:,:,2]
  y_conv1 = get_reg_linear(X, Y, Z)
  y_conv2 = get_reg_linear(X, Z, Y)
  y_conv3 = get_reg_linear(Y, Z, X)
  # y_conv1=tf.square(y_conv1, name=None)
  # y_conv2=tf.square(y_conv2, name=None)
  # y_conv3=tf.square(y_conv3, name=None)

  # y_conv=tf.concat([y_conv1,y_conv2,y_conv3],axis=0)
  # y_conv=tf.reduce_mean(y_conv,axis=0)
  # 損失関数（交差エントロピー誤差）
  # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
  # 勾配
  def loss_fun(y_lable,y_conv):
      y_conv=tf.reshape(y_conv,(1,3))
      loss=tf.subtract(y_lable, y_conv, name=None)
      loss2=tf.reduce_sum(tf.square(loss, name=None))
      return loss2
  loss1=loss_fun(y_,y_conv1)
  loss2=loss_fun(y_,y_conv2)
  loss3=loss_fun(y_,y_conv3)
  loss=tf.add(loss1,loss2)
  loss=tf.add(loss,loss3)
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

  # 精度
  correct_prediction = tf.equal(tf.argmax(y_conv1, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # セッション
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  # トレーニング
  for i in range(50000):
    if i % 100 == 0:
      # 途中経過（500件ごと）
      train_accuracy= accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
      print("step %d, training accuracy %f" % (i, train_accuracy))
      if train_accuracy>0.999:
          W_,y_con1,y_con2,y_con3=sess.run([W_fc2,y_conv1,y_conv2,y_conv3],feed_dict={x: batch_x, y_: batch_y})
          print(y_con1[0])
          print(y_con2[0])
          print(y_con3[0])
          print(W_)

    # トレーニング実行
    train_step.run(feed_dict={x: batch_x, y_: batch_y})

if __name__ == '__main__':
    p = 'C:/00_work/05_src/data/frm_t/20201015155835'
    f = f"{p}/pcd_extracted.ply"
    pcd = open3d.io.read_point_cloud(f)
    points = np.copy(np.array(pcd.points))
    batch_x=[points]

    colors = np.copy(np.array(pcd.colors))
    bt_size=1
    #v_dst=[0,1,0]
    batch_y=np.array([[0,0,1]])
    main(batch_x,batch_y,bt_size)