(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/LineSentenceIterator.html"}
  dl4clj.text.sentenceiterator.line-sentence-iterator
  (require [dl4clj.text.sentenceiterator.sentence-iterator :refer :all])
  (:import [org.deeplearning4j.text.sentenceiterator LineSentenceIterator]
           [java.io InputStream]))

(defn line-sentence-iterator [x]
  (LineSentenceIterator. ^java.io.File (clojure.java.io/as-file x)))

(defmethod next-sentence LineSentenceIterator [^LineSentenceIterator i] 
  (.nextSentence i))

(defmethod has-next LineSentenceIterator [^LineSentenceIterator i] 
  (.hasNext i))

(defmethod reset LineSentenceIterator [^LineSentenceIterator i]
  (.reset i))

(defmethod finish LineSentenceIterator [^LineSentenceIterator i]
  (.finish i))

(defmethod get-pre-processor LineSentenceIterator [^LineSentenceIterator i]
  (.getPreProcessor i))

(defmethod set-pre-processor LineSentenceIterator [^LineSentenceIterator i p]
  (.setPreProcessor i p))
