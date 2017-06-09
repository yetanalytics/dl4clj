(ns dl4clj.nn.conf.graph.graph-vertex
  (:import [org.deeplearning4j.nn.graph.vertex
            VertexIndices BaseGraphVertex GraphVertex]))
(comment

  (defprotocol graph-vertex-interface
  "an implementation of the GraphVertex interface, see
  https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/GraphVertex.html"
  (can-do-backwards [this ^boolean x]
    "Whether the GraphVertex can do backward pass. Typically, this is just whether all errors/epsilons are set")
  (can-do-forward [this ^boolean x]
    "Whether the GraphVertex can do forward pass. Typically, this is just whether all inputs are set.")
  (clear [this] "Clear the internal state (if any) of the GraphVertex. For example, any stored inputs/errors")
  (do-backwards [this ^boolean tbptt] "Do backward pass, tbptt - If true: do backprop using truncated BPTT")
  (do-forward [this ^boolean training] "Do forward pass using the stored inputs,
   training - if true: forward pass at training time. If false: forward pass at test time")
  (get-errors [this] "Get the array of errors previously set for this GraphVertex")
  (get-inputs [this] "Get the array of inputs previously set for this GraphVertex")
  (get-input-vertices [this] "A representation of the vertices that are inputs to
   this vertex (inputs duing forward pass)  Specifically, if
   inputVertices[X].getVertexIndex() = Y, and inputVertices[X].getVertexEdgeNumber() = Z
   then the Zth output connection (see getNumOutputConnections() of vertex Y
   is the Xth input to this vertex")
  (get-layer [this] "Get the Layer (if any). Returns null if hasLayer() == false")
  (get-num-input-arrays [this] "Get the number of input arrays. For example,
    a Layer may have only one input array, but in general a GraphVertex may have
    an arbtrary (>=1) number of input arrays (for example, from multiple other layers)")
  (get-num-output-connections [this] "Get the number of outgoing connections from this GraphVertex.
   A GraphVertex may only have a single output (for example, the activations out of a layer),
   but this output may be used as the input to an arbitrary number of other GraphVertex instances.
   This method returns the number of GraphVertex instances the output of this GraphVertex is input for.")
  (get-output-vertices [this] "A representation of the vertices that this vertex is connected to
  (outputs duing forward pass) Specifically, if outputVertices[X].getVertexIndex() = Y,
  and outputVertices[X].getVertexEdgeNumber() = Z then the Xth output of this vertex
  is connected to the Zth input of vertex Y")
  (get-vertex-index [this] "Get the index of the GraphVertex")
  (get-vertex-name [this] "Get the name/label of the GraphVertex")
  (has-layer [this] "whether the GraphVertex contains a Layer object or not")
  (is-input-vertex [this] "Whether the GraphVertex is an input vertex")
  (is-output-vertex [this] "Whether the GraphVertexis an output vertex")
  (set-backprop-gradients-view-array [this ^INDArray backprop-gradients-view-array]
    "Set the gradients array as a view of the full (backprop) network parameters
    NOTE: this is intended to be used internally in MultiLayerNetwork and ComputationGraph, not by users.")
  (set-error [this ^Integer error-num ^INDArray error] "Set the errors (epsilons) for this GraphVertex")
  (set-errors [this errors] "Set all errors/epsilons for this GraphVertex,
   errors is an INDArray of [int INDArray (error)]")
  (set-input [this ^Integer input-num ^INDArray input] "Set the input activations. params:
   inputNumber - Must be in range 0 to getNumInputArrays()-1, input - The input array")
  (set-inputs [this ^INDArray inputs] "Set all inputs for this GraphVertex,
   coll of set-input params")
  (set-input-vertices [this input-vertices] "Sets the input vertices.
   input-vertices: (into-array VertexIndices)")
  (set-output-vertices [this output-vertices] "set the output vertices.
   :output-vertices (into-array VertexIndices)")
  )


  (extend-type ComputationGraph ;;something like this
    graph-vertex
    ;; methods
    ))
