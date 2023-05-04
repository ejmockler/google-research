from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import tensorflow_addons as tfa

# MultiHeadAttention class implementation (supports cross-attention)
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        # Define dense layers for query, key, and value projections
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        # Define dense layer for output
        self.dense = tf.keras.layers.Dense(d_model)
        
    @property
    def trainable_variables(self):
        variables = []
        variables.extend(self.wq.trainable_variables)
        variables.extend(self.wk.trainable_variables)
        variables.extend(self.wv.trainable_variables)
        variables.extend(self.dense.trainable_variables)
        return variables

    # Split heads in multihead attention
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # Main call method for multihead attention
    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        # Project inputs (query, key, value) using dense layers
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Calculate scaled dot-product attention and attention weights
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

        # Concatenate the multi-head outputs
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, self.d_model))

        # Project the concatenated outputs
        output = self.dense(concat_attention)

        return output, attention_weights

def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

# Modified TabNet (adapted from the Google Research Authors)
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

"""TabNet model."""
def glu(act, n_units):
  """Generalized linear unit nonlinear activation."""
  return act[:, :n_units] * tf.nn.sigmoid(act[:, n_units:])
    
class TabNetMultiHead(tf.keras.Model):
  """TabNet model class."""

  def __init__(self,
               columns,
               num_features,
               feature_dim,
               output_dim,
               num_decision_steps,
               relaxation_factor,
               batch_momentum,
               virtual_batch_size,
               num_classes,
               num_heads=2,
               grouped_feature_columns=None,
               epsilon=0.00001):
    """Initializes a TabNet instance.
    Args:
      columns: The Tensorflow column names for the dataset.
      num_features: The number of input features (i.e the number of columns for
        tabular data assuming each feature is represented with 1 dimension).
      feature_dim: Dimensionality of hidden representation for the peptide embedding 
        for multi-headed attention. During feature transformation, each layer first
        maps the representation to a 2*feature_dim-dimensional output and half 
        of it is used to determine the nonlinearity of the GLU activation where the
        other half is used as an input to GLU, and eventually feature_dim-dimensional 
        output is transferred to the next layer. 
      output_dim: Dimensionality of the outputs of each decision step, which is
        later mapped to the final classification or regression output.
      num_decision_steps: Number of sequential decision steps.
      relaxation_factor: Relaxation factor that promotes the reuse of each
        feature at different decision steps. When it is 1, a feature is enforced
        to be used only at one decision step and as it increases, more
        flexibility is provided to use a feature at multiple decision steps.
      batch_momentum: Momentum in ghost batch normalization.
      virtual_batch_size: Virtual batch size in ghost batch normalization. The
        overall batch size should be an integer multiple of virtual_batch_size.
      num_classes: Number of output classes.
      num_heads: Number of heads for multi-headed attention.
      grouped_feature_columns: Column lists to aggregate in the model as a feature group.
      epsilon: A small number for numerical stability of the entropy calcations.
    Returns:
      A TabNet instance.
    """
    super().__init__()
    self.columns = columns
    self.num_features = num_features
    self.feature_dim = feature_dim
    self.output_dim = output_dim
    self.num_decision_steps = num_decision_steps
    self.relaxation_factor = relaxation_factor
    self.batch_momentum = batch_momentum
    self.virtual_batch_size = virtual_batch_size
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.num_heads = num_heads
    self.missing_feature_params = self.add_missing_feature_params()
    self.grouped_feature_columns = grouped_feature_columns
    self.feature_columns_dict = {col.key: col for col in self.columns}
    
    self.input_layer = tf.keras.layers.DenseFeatures(feature_columns=self.columns)
    self.multi_attention_layer = self.multiAttention = MultiHeadAttention(feature_dim, num_heads)
    
    self.classify_layer = tf.keras.layers.Dense(self.num_classes)
    self.regress_layer = tf.keras.layers.Dense(self.num_classes)


  def add_missing_feature_params(self):
    missing_feature_params = []
    for i in range(self.num_features):
        missing_param = tf.Variable(initial_value=tf.zeros(shape=(1,), dtype=tf.float32), trainable=True, name='missing_param_'+str(i))
        missing_feature_params.append(missing_param)
    return missing_feature_params
  
  def call(self, data, peptide_embeddings, is_training, peptide_column_group_index=None):
        output_aggregated, total_entropy = self.encoder(data, peptide_embeddings, is_training, peptide_column_group_index)
        logits, predictions = self.classify(output_aggregated)
        regression_output = self.regress(output_aggregated)

        return logits, predictions, regression_output, total_entropy

  def encoder(self, data, peptide_embeddings, is_training, peptide_column_group_index=None):
    """TabNet encoder model."""

    with tf.name_scope("Encoder"):
        # Reads and normalizes input features.
        if self.grouped_feature_columns is None:
            features = self.input_layer(data)
        else:
            grouped_features = []
            for i, group_columns in enumerate(self.grouped_feature_columns):
                group_feature_columns = [self.feature_columns_dict[col_name] for col_name in group_columns]
                group_input_layer = tf.keras.layers.DenseFeatures(feature_columns=group_feature_columns)
                group_features = group_input_layer(data)

                # Attach peptide embedding to the specified column group index
                if peptide_column_group_index is not None and i == peptide_column_group_index:
                    group_features = tf.concat([group_features, peptide_embeddings], axis=-1)

                grouped_features.append(group_features)
            features = tf.concat(grouped_features, axis=-1)
        batch_size, self.num_features = tf.shape(features)
        
        # Replace missing values with trainable parameters
        mask = tf.math.is_nan(features)
        missing_feature_params_expanded = tf.reshape(self.missing_feature_params, (1, self.num_features))
        missing_feature_params_tiled = tf.tile(missing_feature_params_expanded, [batch_size, 1])
        features = tf.where(mask, missing_feature_params_tiled, features)
        features = tf.keras.layers.BatchNormalization(momentum=self.batch_momentum)(features, training=is_training)
        
        # Apply MultiHeadAttention on the input features and peptide embeddings
        attention_output, _ = self.multi_attention_layer(peptide_embeddings, peptide_embeddings, features)

        # Concatenate the attention_output with the input features
        reshaped_attention_output = tf.reshape(attention_output, (batch_size, self.feature_dim))
        masked_features = tf.concat([features, reshaped_attention_output], axis=1)
        
        # Expand into table
        projection_layer_table = tf.keras.layers.Dense(self.num_features)
        weighted_table = projection_layer_table(masked_features)

        # Initializes decision-step dependent variables.
        output_aggregated = tf.zeros([batch_size, self.output_dim])
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complementary_aggregated_mask_values = tf.ones([batch_size, self.num_features])
        total_entropy = 0

        if is_training:
            v_b = self.virtual_batch_size
        else:
            v_b = 1
        
        # Expand into peptide
        projection_layer_peptide = tf.keras.layers.Dense(peptide_embeddings.shape[-1])
        weighted_peptide = projection_layer_peptide(masked_features)
        
       # Apply stacked MultiHeadAttention on the weighted_peptide and weighted_table
        attention_output_integrated = tf.expand_dims(weighted_table, axis=1)
        attention_output_integrated, _ = self.multi_attention_layer(weighted_peptide, peptide_embeddings, attention_output_integrated)

        projection_layer_transform = tf.keras.layers.Dense(self.feature_dim)
        masked_features = projection_layer_transform(attention_output_integrated)
        
        for ni in range(self.num_decision_steps):

            # Feature transformer with two shared and two decision step dependent
            # blocks is used below.

            transform_f1 = tf.keras.layers.Dense(
                self.feature_dim * 2,
                name="Transform_f1",
                use_bias=False)(masked_features)
            transform_f1 = tf.keras.layers.BatchNormalization(momentum=self.batch_momentum)(transform_f1, training=is_training)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = tf.keras.layers.Dense(
                self.feature_dim * 2,
                name="Transform_f2",
                use_bias=False)(transform_f1)
            transform_f2 = tf.keras.layers.BatchNormalization(momentum=self.batch_momentum)(transform_f2, training=is_training)
            transform_f2 = (glu(transform_f2, self.feature_dim) + transform_f1) * np.sqrt(0.5)

            transform_f3 = tf.keras.layers.Dense(
                self.feature_dim * 2,
                name="Transform_f3" + str(ni),
                use_bias=False)(transform_f2)
            transform_f3 = tf.keras.layers.BatchNormalization(
                momentum=self.batch_momentum,
                virtual_batch_size=v_b)(transform_f3, training=is_training)
            transform_f3 = (glu(transform_f3, self.feature_dim) +
                            transform_f2) * np.sqrt(0.5)

            transform_f4 = tf.keras.layers.Dense(
                self.feature_dim * 2,
                name="Transform_f4" + str(ni),
                use_bias=False)(transform_f3)
            transform_f4 = tf.keras.layers.BatchNormalization(
                momentum=self.batch_momentum,
                virtual_batch_size=v_b)(transform_f4, training=is_training)
            transform_f4 = (glu(transform_f4, self.feature_dim) +
                            transform_f3) * np.sqrt(0.5)

            if ni > 0:

                decision_out = tf.keras.activations.relu(transform_f4[:, :self.output_dim])

                # Decision aggregation.
                output_aggregated += decision_out

                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = tf.reduce_sum(
                    decision_out, axis=1, keepdims=True) / (
                        self.num_decision_steps - 1)
                aggregated_mask_values += mask_values * scale_agg

            features_for_coef = (transform_f4[:, self.output_dim:])

            if ni < self.num_decision_steps - 1:

                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use.
                mask_values = tf.keras.layers.Dense(
                    self.num_features,
                    name="Transform_coef" + str(ni),
                    use_bias=False)(features_for_coef)
                mask_values = tf.keras.layers.BatchNormalization(
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b)(mask_values, training=is_training)
                mask_values *= complementary_aggregated_mask_values
                mask_values = tfa.layers.Sparsemax()(mask_values)

                # Relaxation factor controls the amount of reuse of features between
                # different decision blocks and updated with the values of
                # coefficients.
                complementary_aggregated_mask_values *= (
                    self.relaxation_factor - mask_values)

                # Entropy is used to penalize the amount of sparsity in feature
                # selection.
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        -mask_values * tf.math.log(mask_values + self.epsilon),
                        axis=1)) / (
                            self.num_decision_steps - 1)

                # Feature selection.
                masked_features = tf.multiply(mask_values, features)

                # Visualization of the feature selection mask at decision step ni
                tf.summary.image(
                    "Mask for step" + str(ni),
                    tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
                    max_outputs=1)


    # Visualization of the aggregated feature importances
    tf.summary.image(
        "Aggregated mask",
        tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
        max_outputs=1)

    return output_aggregated, total_entropy

  def classify(self, activations):
    logits = self.classify_layer(activations)
    predictions = tf.nn.softmax(logits)
    return logits, predictions

  def regress(self, activations):
    logits = self.regress_layer(activations)
    return logits
