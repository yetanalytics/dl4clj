(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/documentiterator/LabelAwareIterator.html"}
    dl4clj.text.documentiterator.label-aware-iterator
  (:import [org.deeplearning4j.text.documentiterator LabelAwareIterator]))

(defn get-labels-source
  ""
  [^LabelAwareIterator this]
  (.getLabelsSource this))

(defn has-next-document
  ""
  [^LabelAwareIterator this]
  (.hasNextDocument this))

(defn next-document
  ""
  [^LabelAwareIterator this]
  (.nextDocument this))

(defn reset
  ""
  [^LabelAwareIterator this]
  (.reset this))

(defn- documents-iter [^LabelAwareIterator this]
  (when (has-next-document this)
    (cons (next-document this)
          (lazy-seq (documents-iter this)))))

(defn documents
  "Returns a lazy sequence of the documents. Not thread safe!"
  [^LabelAwareIterator this]
  (reset this)
  (documents-iter this))
