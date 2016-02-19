(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/WeightLookupTable.html"}
  dl4clj.models.embeddings.weight-lookup-table
  (:import [org.deeplearning4j.models.embeddings WeightLookupTable]))

(defn get-gradient
  "Returns gradient for specified word"
  [^WeightLookupTable this column gradient]
  (.getGradient this (int column) (double gradient)))

(defn get-table-id
  "Returns unique ID of this table."
  [^WeightLookupTable this]
  (.getLabelId this))

(defn get-vocab-cache
  "Returns corresponding vocabulary"
  [^WeightLookupTable this]
  (.getVocabCache this))

(defn get-weights
  [^WeightLookupTable this]
  (.getWeights this))

(defn layer-size
  "The layer size for the lookup table"
  [^WeightLookupTable this]
  (.layerSize this))

(defn load-codes 
  "Loads the co-occurrences for the given codes"
  [^WeightLookupTable this ints]
  (.loadCodes this (int-array ints)))

(defn plot-vocab
  "Render the words via tsne"
  ([^WeightLookupTable this]
   (.plotVocab this))
  ([^WeightLookupTable this tsne]
   (.plotVocab this tsne)))

(defn put-code 
  [^WeightLookupTable this code-index code]
  (.putCode this (int code-index) code))

(defn put-vector
  "Inserts a word vector"
  [^WeightLookupTable this ^String word vector]
  (.putVector this word vector))

(defn reset-weights
  "Reset the weights of the cache"
  ([^WeightLookupTable this]
   (.resetWeights this))
  ([^WeightLookupTable this reset?]
   (.resetWeights this (boolean reset?))))

(defn set-learning-rate
  "Sets the learning rate"
  [^WeightLookupTable this lr]
  (.setLearningRate this (double lr)))

(defn set-table-id 
  "Set's table Id."
  [^WeightLookupTable this table-id]
  (.setTableId this (long table-id)))

(defn vector
  [^WeightLookupTable this ^String word]
  (.vector this word))

(defn vectors
  "Iterates through all of the vectors in the cache"
  [^WeightLookupTable this]
  (.vectors this))
