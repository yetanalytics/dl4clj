(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/SentenceIterator.html"}
  dl4clj.text.sentenceiterator.sentence-iterator
  (:import [org.deeplearning4j.text.sentenceiterator SentenceIterator]
           [org.deeplearning4j.text.sentenceiterator SentencePreProcessor]))

(defn next-sentence 
  "Gets the next sentence or null if there's nothing left (Do yourself a favor and check has-next)" 
  [^SentenceIterator i]
  (.nextSentence i))

(defn has-next 
  "whether there's anymore sentences left"
  [^SentenceIterator i]
  (.hasNext i))

(defn reset 
  "Resets the iterator to the beginning"
  [^SentenceIterator i]
  (.reset i))

(defn finish 
  "Allows for any finishing (closing of input streams or the like)"
  [^SentenceIterator i]
  (.finish i))

(defn get-pre-processor 
  [^SentenceIterator i]
  (.getPreProcessor i))

(defn set-pre-processor 
  [^SentenceIterator i ^SentencePreProcessor p]
  (.setPreProcessor i p))


