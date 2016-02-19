(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/LineSentenceIterator.html"}
  dl4clj.text.sentenceiterator.line-sentence-iterator
  (:import [org.deeplearning4j.text.sentenceiterator LineSentenceIterator]
           [java.io InputStream]))

(defn line-sentence-iterator [x]
  (LineSentenceIterator. ^java.io.File (clojure.java.io/as-file x)))

