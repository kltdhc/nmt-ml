# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
import tensorflow as tf


__all__ = [
    "BasicDecoderOutput",
    "BasicDecoder",
]


class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
  pass


class BasicDecoder(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(self, cells, helper, initial_states, output_layers=None):
    """Initialize BasicDecoder.

    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Optional layer to apply to the RNN output prior
        to storing the result or sampling.

    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    for i in range(len(cells)):
        # rnn_cell_impl.assert_like_rnncell("cell[%d]"%i, cells[i])
        if (output_layers is not None
            and not isinstance(output_layers[i], layers_base.Layer)):
            raise TypeError(
                "output_layer[%d] must be a Layer, received: %s" % (i, type(output_layers[i])))
    if not isinstance(helper, helper_py.Helper):
        raise TypeError("helper must be a Helper, received: %s" % (type(helper)))
    self._cells = cells
    self._helper = helper
    self._initial_states = initial_states
    self._output_layers = output_layers

  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._cells[0].output_size
    if self._output_layers is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_layers[0].compute_output_shape(
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        rnn_output=self._rnn_output_size(),
        sample_id=self._helper.sample_ids_shape)

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and the sample_ids_dtype from the helper.
    dtype = nest.flatten(self._initial_states[0])[0].dtype
    return BasicDecoderOutput(
        nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        self._helper.sample_ids_dtype)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    # rt = []
    # for i in range(len(self._initial_states)):
    #     rt.append(self._helper.initialize() + (self._initial_states[i],))
    return self._helper.initialize() + (self._initial_states,)

  def step(self, time, inputs, states, name=None):
      """Perform a decoding step.

      Args:
        time: scalar `int32` tensor.
        inputs: A (structure of) input tensors.
        state: A (structure of) state tensors and TensorArrays.
        name: Name scope for any created operations.

      Returns:
        `(outputs, next_state, next_inputs, finished)`.
      """
      outputs = []
      nstates = []
      with tf.variable_scope("MultiDecoderScope"):
          for i in range(len(self._cells)):
              cell_outputs, cell_state = self._cells[i](inputs, states[i])
              if self._output_layers is not None:
                  cell_outputs = self._output_layers[i](cell_outputs)
              outputs.append(cell_outputs)
              nstates.append(cell_state)
          outputs = tf.add_n(outputs)
          sample_ids = self._helper.sample(time=time, outputs=outputs, state=cell_state)
          (finished, next_inputs, next_state) = self._helper.next_inputs(
              time=time,
              outputs=outputs,
              states=nstates,
              sample_ids=sample_ids)
          outputs = BasicDecoderOutput(outputs, sample_ids)
      return (outputs, next_state, next_inputs, finished)