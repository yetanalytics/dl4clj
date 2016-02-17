(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/BasiclineIterator.html"}
  dl4clj.text.sentenceiterator.basic-line-iterator
  (require [dl4clj.text.sentenceiterator.sentence-iterator :refer :all])
  (:import [org.deeplearning4j.text.sentenceiterator BasicLineIterator]
           [java.io InputStream]))

(defn basic-line-iterator [x]
  (BasicLineIterator. ^InputStream (clojure.java.io/input-stream x)))

(defmethod next-sentence BasicLineIterator [^BasicLineIterator i] 
  (.nextSentence i))

(defmethod has-next BasicLineIterator [^BasicLineIterator i] 
  (.hasNext i))

(defmethod reset BasicLineIterator [^BasicLineIterator i]
  (.reset i))

(defmethod finish BasicLineIterator [^BasicLineIterator i]
  (.finish i))

(defmethod get-pre-processor BasicLineIterator [^BasicLineIterator i]
  (.getPreProcessor i))

(defmethod set-pre-processor BasicLineIterator [^BasicLineIterator i p]
  (.setPreProcessor i p))


