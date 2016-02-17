(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/WeightLookupTable.html"}
  dl4clj.models.embeddings.weight-lookup-table
  (:import [org.deeplearning4j.models.embeddings WeightLookupTable]))

(defmulti get-gradient
  "Returns gradient for specified word"
  (fn [this column gradient] (type this)))

(defmulti get-table-id
  "Returns unique ID of this table."
  (fn [this] (type this)))

(defmulti get-vocab-cache
  "Returns corresponding vocabulary"
  (fn [this] (type this)))

(defmulti get-weights
  (fn [this] (type this)))

(defmulti layer-size
  "The layer size for the lookup table"
  (fn [this] (type this)))

(defmulti load-codes 
  "Loads the co-occurrences for the given codes"
  (fn [this ints] (type this)))

(defmulti plot-vocab
  "Render the words via tsne"
  (fn [this & more ] (type this)))

(defmulti put-code 
  (fn [this code-index code] (type this)))

(defmulti put-vector
  "Inserts a word vector"
  (fn [this word vector] (type this)))

(defmulti reset-weights
  "Reset the weights of the cache"
  (fn [this & more] (type this)))

(defmulti set-learning-rate
  "Sets the learning rate"
  (fn [this lr] (type this)))

(defmulti set-table-id 
  "Set's table Id."
  (fn [this table-id] (type this)))

(defmulti vector
  (fn [this word] (type this)))

(defmulti vectors
  "Iterates through all of the vectors in the cache"
  (fn [this] (type this)))
