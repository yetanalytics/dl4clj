(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/inmemory/InMemoryLookupTable.html"}
  dl4clj.models.embeddings.inmemory.in-memory-lookup-table
  (:require [dl4clj.models.embeddings.weight-lookup-table :refer :all])
  (:import [org.deeplearning4j.models.embeddings.inmemory InMemoryLookupTable]
           [org.deeplearning4j.plot Tsne]))

(defmethod get-gradient InMemoryLookupTable
  [^InMemoryLookupTable this column gradient]
  (.getGradient this (int column) (double gradient)))

(defmethod get-table-id InMemoryLookupTable
  [^InMemoryLookupTable this]
  (.getTableId this))

(defmethod get-vocab-cache InMemoryLookupTable
  [^InMemoryLookupTable this] 
  (.getVocabCache this))

(defmethod get-weights InMemoryLookupTable
  [^InMemoryLookupTable this]
  (.getWeights this))

(defmethod layer-size InMemoryLookupTable
  [^InMemoryLookupTable this]
  (.layerSize this))

(defmethod load-codes InMemoryLookupTable
  [^InMemoryLookupTable this ints]
  (.loadCodes this (int-array ints)))

(defmethod plot-vocab InMemoryLookupTable
  ([^InMemoryLookupTable this]
   (.plotVocab this))
  ([^InMemoryLookupTable this ^Tsne tsne]
   (.plotVocab this tsne)))

(defmethod put-code InMemoryLookupTable
  [^InMemoryLookupTable this code-index code]
  (.putCode this (int code-index) code))

(defmethod put-vector InMemoryLookupTable
  [^InMemoryLookupTable this ^String word vector]
  (.putVector this word vector))

(defmethod reset-weights InMemoryLookupTable
  ([^InMemoryLookupTable this]
   (.resetWeights this))
  ([^InMemoryLookupTable this reset]
   (.resetWeights this (boolean this))))

(defmethod set-learning-rate InMemoryLookupTable
  [^InMemoryLookupTable this lr]
  (.setLearningRate this (double lr)))

(defmethod set-table-id InMemoryLookupTable
  [^InMemoryLookupTable this table-id]
  (.setTableId this (long table-id)))

(defmethod vector InMemoryLookupTable
  [^InMemoryLookupTable this ^String word]
  (.vector this word))

(defmethod vectors InMemoryLookupTable
  [^InMemoryLookupTable this]
  (.vectors this))
