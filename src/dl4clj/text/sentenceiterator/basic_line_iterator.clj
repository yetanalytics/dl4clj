(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/BasiclineIterator.html"}
  dl4clj.text.sentenceiterator.basic-line-iterator
  (:import [org.deeplearning4j.text.sentenceiterator BasicLineIterator]
           [java.io InputStream]))

(defn basic-line-iterator [x]
  (BasicLineIterator. ^InputStream (clojure.java.io/input-stream x)))
