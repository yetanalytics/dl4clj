(ns dl4clj.nn.conf.builders.vertex
  (:import #_[org.deeplearning4j.nn.graph.vertex
            VertexIndices BaseGraphVertex]
           [org.deeplearning4j.nn.conf ComputationGraphConfiguration$GraphBuilder
            ComputationGraphConfiguration
            NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.graph ComputationGraph]
           [org.deeplearning4j.nn.graph.vertex.impl
            PreprocessorVertex ElementWiseVertex SubsetVertex
            MergeVertex LayerVertex InputVertex ]
           [org.deeplearning4j.nn.graph.vertex.impl.rnn
            DuplicateToTimeSeriesVertex LastTimeStepVertex]
           [org.deeplearning4j.nn.conf.graph GraphVertex] ;; this is the class
           [org.deeplearning4j.berkeley Pair] ;; might not need this
           ))
(comment
  (GraphVertex. {}) ;; this says it needs a constructor when doc says it doesnt????
  ;; still busted even tho now im only requring the class not the interface

  ;;need a multimethod for setting up the class GraphVertex


  ;;https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/ComputationGraph.html
  ;; constructor just needs a ComputationGraphConfiguration
  (ComputationGraph. (ComputationGraphConfiguration.))


  ;; being weird https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/impl/ElementWiseVertex.Op.html

  ;; needs ComputationGraph, name (str), idx (int), input-vertices (into-array VertexIndices)
  ;; output-verticies (into-array VertexIndices)
  ;; vertexindices https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/VertexIndices.html
  (BaseGraphVertex.)

  ;;https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/VertexIndices.html
  ;; docs says constructor doesnt need anything but (VertexIndices.) says no matching ctor
  (VertexIndices.)

  ;;https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/impl/PreprocessorVertex.html
  ;; needs ComputationGraph, name (str), idx (int),
  ;;[these 2 are optional: input-verticies and output-vertices] (into-array Vertexindices)
  ;;InputPreProcessor (map {:input-pre-processor-type opts})
  (PreprocessorVertex.)

  ;;https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/impl/ElementWiseVertex.html
  ;; has ElementWiseVertex.Op but this is acting weird
  ;; constructor needs ComputationGraph, name (str), vertex-idx (int), ElementWiseVertex.Op
  ;; or ComputationGraph, name (str), vertex-idx (int), (into-array Vertex-indices) (input and output), ElementWiseVertex.op
  (ElementWiseVertex.)

  ;;https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/impl/SubsetVertex.html
  ;; constructor needs ComputationGraph, name (str), vertex-idx (int), from (int), to (int)
  ;; or ComputationGraph, name (str), vertex-idx (int) (into-array VertexIndices [input and output]), from (int), to (int)
  (SubsetVertex.)

  ;;https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/impl/MergeVertex.html
  ;; constructor needs ComputationGraph, name (str), vertex-idx (int)
  ;; or ComputationGraph, name (str), vertex-idx (int), (into-array VertexIndices [input and output]),
  (MergeVertex.)

  ;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/impl/LayerVertex.html
  ;; constructor needs ComputationGraph, name (str), vertex-idx (int), InputPreProcessor (map), output-vertex (boolean)
  ;; or ComputationGraph, name (str), vertex-idx (int), (into-array VertexIndices [input and output]), layer (map or fn),
  ;; InputPreProcessor (map), output-vertex (boolean)
  (LayerVertex.)

  ;;https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/impl/InputVertex.html
  ;; constructor needs ComputationGraph, name (str), vertex-idx (int), (into-array VertexIndices [only output])
  (InputVertex.)

  ;;https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/impl/rnn/DuplicateToTimeSeriesVertex.html
  ;; constructor needs ComputationGraph, name (str), vertex-idx (int), input-vertex-name (str)
  ;; or ComputationGraph, name (str), vertex-idx (int), (into-array VertexIndices [input and output]), input-name (str)
  (DuplicateToTimeSeriesVertex. (.build (ComputationGraphConfiguration$GraphBuilder. (NeuralNetConfiguration$Builder.))) "foo" 0 "baz")

  ;;https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/vertex/impl/rnn/LastTimeStepVertex.html
  ;; constructor needs ComputationGraph, name (str), vertex-idx (int), input-name (str)
  ;; or ComputationGraph, name (str), vertex-idx (int), (into-array VertexIndices [input and output]), input-name (str)
  (LastTimeStepVertex.)
  )
