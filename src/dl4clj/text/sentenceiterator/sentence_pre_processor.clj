(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/SentencePreProcessor.html"}
  dl4clj.text.sentenceiterator.sentence-pre-processor
  (:import [org.deeplearning4j.text.sentenceiterator SentencePreProcessor]))

(defmulti pre-process 
  (fn [^SentencePreProcessor p ^String s] (type p)))

(deftype CustomSentencePreprocessor [preprocessor]
  SentencePreProcessor
  (preProcess [this s]
    (preprocessor s)))

(defmethod pre-process CustomSentencePreprocessor [p s]
  (.preProcess ^SentencePreProcessor p s))

(defn sentence-pre-processor [preprocessor]
  (CustomSentencePreprocessor. preprocessor))

(comment
  
  (def p (sentence-preprocessor clojure.string/lower-case))

  (pre-process p "Foo Bar !")

  )
