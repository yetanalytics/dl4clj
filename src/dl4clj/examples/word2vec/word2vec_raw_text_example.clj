(ns ^{:doc "Deeplearning4j Word2Vec example based on http://deeplearning4j.org/word2vec.html#just"}
  Dl4clj.examples.word2vec.word2vec-raw-text-example
  (:require [dl4clj.examples.example-utils :refer (shakespeare-file)]
            [dl4clj.text.sentenceiterator.basic-line-iterator :refer (basic-line-iterator)]
            [dl4clj.text.tokenization.tokenizerfactory.tokenizer-factory :refer (set-token-pre-processor)]
            [dl4clj.text.tokenization.tokenizerfactory.default-tokenizer-factory :refer (default-tokenizer-factory)]
            [dl4clj.text.tokenization.tokenizer.token-pre-process :refer (token-pre-process pre-process)]
            [dl4clj.text.sentenceiterator.sentence-pre-processor :refer (sentence-pre-processor)]
            [dl4clj.text.sentenceiterator.sentence-iterator :refer (set-pre-processor)]
            [dl4clj.text.sentenceiterator.line-sentence-iterator :refer (line-sentence-iterator)]
            [dl4clj.text.tokenization.tokenizer.preprocessor.ending-pre-processor :refer (ending-pre-processor)]
            [dl4clj.models.sequencevectors.sequence-vectors :refer (fit)]
            [dl4clj.models.embeddings.wordvectors.word-vectors :refer (similarity words-nearest get-word-vector)]
            [dl4clj.models.word2vec.word2vec :refer (word2vec)]
            [dl4clj.plot.barnes-hut-tsne :refer (barnes-hut-tsne)]
            [dl4clj.models.embeddings.loader.word-vector-serializer :refer (write-word-vectors load-txt-vectors)]))

;;; This example code follows the word2vec code snippets at http://deeplearning4j.org/word2vec.html#just

;;; Loading data

(def iter (line-sentence-iterator (clojure.java.io/resource "raw_sentences.txt")))
(set-pre-processor iter (sentence-pre-processor clojure.string/lower-case))

;;; Tokenizing the data

(def tokenizer (default-tokenizer-factory))
(def ending-pp (ending-pre-processor))
(def custom-pp (token-pre-process (fn [^String token]
                                    (let [base (clojure.string/replace
                                                (pre-process ending-pp
                                                             (clojure.string/lower-case token))
                                                "\\d" "d")]
                                      (when (or (.endsWith base "ly") (.endsWith base "ing"))
                                        (println))
                                      base))))
(set-token-pre-processor tokenizer custom-pp)

;;; Training the model

(def v (-> (word2vec {:batch-size 1000 ;; words per minibatch.
                      :min-word-frequency 5
                      :use-ada-grad false
                      :layer-size 50 ;; word feature vector size
                      :iterations 3  ;; iterations to train
                      :learningRate 0.025
                      :min-learning-rate 1e-3 ;; learning rate decays wrt # words. floor learning
                      :negative-sample 10     ;; sample size 10 words
                      :iterate iter
                      :tokenizer-factory tokenizer})
           (fit)))

;;; Evalating the model

(similarity v "people" "money")
;; => 0.22
(similarity v "day" "night")
;; => 0.70

(words-nearest v ["man"] [] 10)
;; => ["president" "team" ":" "program" "fami" "company" "law" "unit" "director" "part"]
(words-nearest v ["fami"] [] 10)
;; => ["life" "own" "children" "business" "house" "team" "country" "company" "part" "home"]
(words-nearest v ["day"] [] 10)
;; => ["year" "week" "night" "game" "season" "ago" "while" "time" "children" "people"]
(words-nearest v ["king" "woman"] ["queen"] 10)
;; => [] ("king" is not in the model)


;;; Visualizing the Model
;; broken
(def tsne (barnes-hut-tsne {:max-iter 1000
                            :stop-lying-iteration 50
                            :learning-rate 500
                            :use-ada-grad false
                            :theta 0.5
                            :momentum 0.5
                            :normalize true
                            :use-pca false}))
;; broken:
;; (plot-vocab (lookup-table v) tsne)

;;; Saving, Reloading & Using the Model

(write-word-vectors v "/tmp/words.txt")
(def word-vectors (load-txt-vectors "/tmp/words.txt"))
(get-word-vector word-vectors "queen")
