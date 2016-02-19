(ns ^{:doc "
A DataSetIterator for use in the graves-lstm-char-modelling-example

@author Joachim De Beule
"}
  dl4clj.examples.rnn.lr-character-iterator
  (:refer-clojure :exclude [next])
  (:require [nd4clj.linalg.factory.nd4j :refer (zeros)]
            [dl4clj.examples.example-utils :refer (shakespeare index-map)]
            [nd4clj.linalg.api.ndarray.indarray :refer (put-scalar get-scalar shape)]
            [nd4clj.linalg.dataset.api.data-set :refer (get-features get-labels)]
            [nd4clj.linalg.dataset.data-set :refer (data-set)])
  (:import [java.util NoSuchElementException]           
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]))

(defrecord LRCharDataSetIterator [valid-characters char-to-idx-map ^"[C" input-chars segment-length n-segments mini-batch-size char-pointer max-char-pointer]
  DataSetIterator
  (batch [this] mini-batch-size)
  (cursor [this] (throw (Exception. "not yet implemented")))
  (inputColumns [this] (throw (Exception. "not yet implemented")))
  (hasNext [this] (<= @char-pointer (- max-char-pointer (* mini-batch-size segment-length))))
  (next [this] (.next this mini-batch-size))
  (next [this n]
    (when  (> (+ @char-pointer (* n segment-length)) max-char-pointer)
      (throw (NoSuchElementException.)))
    ;; Allocate space:
    (let [input (zeros [n (count valid-characters) segment-length])
          labels (zeros [n (count valid-characters) segment-length])]
      (dotimes [i n]
        (let [start-idx @char-pointer
              end-idx (swap! char-pointer + segment-length)]
          (put-scalar input [i (char-to-idx-map (aget input-chars start-idx)) 0] 1.0)
          (doseq [c (range 1 segment-length)]
            (let [char-idx (char-to-idx-map (aget input-chars (+ start-idx c)))]
              (put-scalar input [i char-idx c] 1.0)
              (put-scalar labels [i char-idx (dec c)] 1.0)))
          (put-scalar labels [i  (char-to-idx-map (aget input-chars end-idx)) (dec segment-length)] 1.0)))
      (data-set input labels)))
  (numExamples [this] n-segments)
  (reset [this] (reset! char-pointer 0))
  (totalExamples [this] n-segments) 
  (totalOutcomes [this] (count valid-characters)))

(defn lr-character-iterator
  ([string]
   (lr-character-iterator string {}))
  ([^String string {:keys [mini-batch-size    ;; number of text segments per batch
                           segment-length ;; number of characters per text segment
                           n-segments ;; number of segments to iterate. Leave unspecified to return the maximum number of segments
                           valid-characters] ;; set of allowed characters. Leave unspecified to allow all characters.
                    :or {mini-batch-size 100
                         segment-length 100}
                    :as opts}]
   (let [valid-characters (into #{} (or valid-characters string))
         input-chars (char-array (if valid-characters (filter valid-characters string) string))
         char-to-idx-map (index-map valid-characters)
         char-pointer (atom 0)
         max-char-pointer (dec (count input-chars))
         max-segments (Math/floorDiv (long max-char-pointer) (long segment-length))]
     (when (and n-segments (> n-segments max-segments)) 
       (throw (IllegalArgumentException. (str "n-segments exceeds number of available segments " max-segments))))
     (when (and n-segments (not (zero? (mod n-segments mini-batch-size))))
       (throw (IllegalArgumentException. (str "n-segments must be a multiple of mini-batch-size"))))
     (println "data has" (count string) "characters," (count valid-characters) "unique.")
     (LRCharDataSetIterator. valid-characters 
                             (index-map valid-characters) 
                             input-chars 
                             segment-length 
                             (or n-segments max-segments)
                             mini-batch-size 
                             (atom 0)
                             max-char-pointer))))

(comment

  ;;; some (inneficient) code for inspecting the examples in an lr-character-iterator

  (require '[nd4j.linalg.dataset.api.iterator.data-set-iterator :refer :all])

  (defn- char-indices [example features-array]
    (for [pos (range (first (shape features-array)))]
      (for [fi (range (second (shape features-array)))]
        (first (get-scalar features-array [example fi pos])))))

  (defn- binary-feature->char-idx [bf]
    (let [idx (remove #(zero? (first %)) (map vector bf (range)))]
      (second (first idx))))

  (defn- batch-examples [batch idx-to-char]
    (for [example (range (nd4clj.linalg.dataset.api.data-set/num-examples batch))]
      (let [features-array (get-features batch)
            labels-array (get-labels batch)]
        {:input (apply str (map #(idx-to-char (binary-feature->char-idx %)) 
                                (char-indices example features-array)))
         :output (apply str (map #(idx-to-char (binary-feature->char-idx %)) 
                                 (char-indices example labels-array)))})))

  (defn example-seq
    "Note: NOT thread safe!"
    [^LRCharDataSetIterator iter]
    (.reset iter)
    (let [idx-to-char (zipmap (map (:char-to-idx-map iter)
                                   (:valid-characters iter))
                              (:valid-characters iter))]
      (mapcat #(batch-examples % idx-to-char) 
              (iterator-seq iter))))


  (def lr-shakespeare-iterator (lr-character-iterator (shakespeare) {}))
  
  (:n-segments lr-shakespeare-iterator)
  ;; => 55898

  (nth (example-seq lr-shakespeare-iterator) 10)
  ;; {:input  "d Shakespeare CDROMS.  Project Gutenberg\r\noften releases Etexts that are NOT placed in the Public Do",
  ;;  :output " Shakespeare CDROMS.  Project Gutenberg\r\noften releases Etexts that are NOT placed in the Public Dom"}

)
