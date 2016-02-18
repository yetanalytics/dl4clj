(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/wordvectors/WordVectorsImpl.html"}
  dl4clj.models.embeddings.wordvectors.word-vectors-impl
  (:require [dl4clj.models.embeddings.wordvectors.word-vectors :refer :all])
  (:import [org.deeplearning4j.models.embeddings.wordvectors WordVectorsImpl]
           [org.nd4j.linalg.api.ndarray INDArray]))

(defn word-vectors-impl []
  (WordVectorsImpl.))

(defmethod accuracy WordVectorsImpl [^WordVectorsImpl impl ^String s]
  (.accuracy impl s))

(defmethod get-word-vector  WordVectorsImpl [^WordVectorsImpl impl ^String s]
  (into [] (.getWordVector impl s)))

(defmethod get-word-vector-matrix WordVectorsImpl [^WordVectorsImpl impl ^String s]
  (.getWordVectorMatrix impl s))

(defmethod get-word-vector-matrix-normalized  WordVectorsImpl [^WordVectorsImpl impl ^String s]
  (.getWordVectorMatrixNormalized impl s))

(defmethod has-word WordVectorsImpl [^WordVectorsImpl impl ^String s]
  (.hasWord impl s))

(defmethod index-of WordVectorsImpl [^WordVectorsImpl impl ^String s]
  (.indexOf impl s))

(defmethod lookup-table WordVectorsImpl [^WordVectorsImpl impl]
  (.lookupTable impl))

(defmethod similarity WordVectorsImpl [^WordVectorsImpl impl ^String s1 ^String s2]
  (.similarity impl s1 s2))

(defmethod similar-words-in-vocab-to  WordVectorsImpl [^WordVectorsImpl impl ^String s ^Double accuracy]
  (.similarWordsInVocabTo impl s accuracy))

(defmethod vocab WordVectorsImpl [^WordVectorsImpl impl]
  (.vocab impl))
        
(defmethod words-nearest WordVectorsImpl 
  ([^WordVectorsImpl impl ^INDArray words ^Number top]
   (.wordsNearest impl words (int top)))
  ([^WordVectorsImpl impl positive-strings negative-strings ^Number top]
   (.wordsNearest impl positive-strings negative-strings (int top))))

(defmethod words-nearest-sum WordVectorsImpl 
  ([^WordVectorsImpl impl ^INDArray words ^Number top]
   (.wordsNearestSum impl words (int top)))
  ([^WordVectorsImpl impl positive-strings negative-strings ^Number top]
   (.wordsNearestSum impl positive-strings negative-strings (int top))))

