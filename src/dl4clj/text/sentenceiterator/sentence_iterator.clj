(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/SentenceIterator.html"}
  dl4clj.text.sentenceiterator.sentence-iterator
  (:import [org.deeplearning4j.text.sentenceiterator SentenceIterator]
           [org.deeplearning4j.text.sentenceiterator SentencePreProcessor]))

(defmulti next-sentence 
  "Gets the next sentence or null if there's nothing left (Do yourself a favor and check has-next)" 
  (fn [^SentenceIterator i] (type i)))

(defmulti has-next 
  "whether there's anymore sentences left"
  (fn [^SentenceIterator i] (type i)))

(defmulti reset 
  "Resets the iterator to the beginning"
  (fn [^SentenceIterator i] (type i)))

(defmulti finish 
  "Allows for any finishing (closing of input streams or the like)"
  (fn [^SentenceIterator i] (type i)))

(defmulti get-pre-processor 
  (fn [^SentenceIterator i] (type i)))

(defmulti set-pre-processor 
  (fn [^SentenceIterator i ^SentencePreProcessor p] (type i)))


