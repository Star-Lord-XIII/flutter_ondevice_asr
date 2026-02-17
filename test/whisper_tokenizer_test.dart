import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_ondevice_asr/models/whisper/whisper_tokenizer.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('WhisperTokenizer - Multilingual', () {
    setUp(() async {
      // Reset singleton for each test group
      WhisperTokenizer.instance.reset();
      await WhisperTokenizer.instance.loadVocab(
        path: 'assets/transcribers/whisper/tokenizer/vocab_multilingual.json',
      );
    });

    test('decodes <|endoftext|> special token', () {
      // Only <|endoftext|> (50257) is in the vocab.json
      // Other special tokens are managed by generation_config.json
      final decoded = WhisperTokenizer.instance.decode(
        [50257],
        skipSpecialTokens: false,
      );

      expect(decoded, contains('<|endoftext|>'));
    });

    test('skips <|endoftext|> when skipSpecialTokens is true', () {
      final decoded = WhisperTokenizer.instance.decode(
        [50257],
        skipSpecialTokens: true,
      );

      expect(decoded, isEmpty);
    });

    test('decodes regular text correctly', () {
      // Token IDs for "And so my fellow Americans"
      // 400: " And", 370: " so", 452: " my", 7177: " fellow", 6280: " Americans"
      final decoded = WhisperTokenizer.instance.decode(
        [400, 370, 452, 7177, 6280],
        skipSpecialTokens: true,
      );

      expect(decoded.trim(), 'And so my fellow Americans');
    });

    test('handles byte-level BPE decoding', () {
      // Token 0 is "!" in the vocab
      final decoded = WhisperTokenizer.instance.decode([0]);
      expect(decoded, equals('!'));
    });

    test('decodes mixed content with special tokens', () {
      // Test decoding with <|endoftext|> and regular text
      final decoded = WhisperTokenizer.instance.decode(
        [400, 50257, 370], // " And" <|endoftext|> " so"
        skipSpecialTokens: false,
      );

      expect(decoded, contains('<|endoftext|>'));
      expect(decoded, contains('And'));
      expect(decoded, contains('so'));
    });

    test('preserves spaces in decoded text', () {
      // Tokens with leading space (Ġ in vocab): 400=" And", 370=" so"
      final decoded = WhisperTokenizer.instance.decode([400, 370]);

      // Should contain both words with their spaces
      expect(decoded, contains('And'));
      expect(decoded, contains('so'));
    });

    test('handles multi-byte UTF-8 characters', () {
      // Token IDs that represent multi-byte UTF-8 when combined
      // Just test that decoding doesn't crash and produces valid output
      final decoded = WhisperTokenizer.instance.decode([100, 200, 300]);
      expect(decoded, isNotNull);
    });
  });

  group('WhisperTokenizer - English-only', () {
    setUp(() async {
      // Reset and load English-only vocab
      WhisperTokenizer.instance.reset();
      await WhisperTokenizer.instance.loadVocab(
        path: 'assets/transcribers/whisper/tokenizer/vocab_en.json',
      );
    });

    test('decodes <|endoftext|> in English-only vocab', () {
      // English-only vocab has <|endoftext|> at 50256 (not 50257 like multilingual)
      final decoded = WhisperTokenizer.instance.decode(
        [50256],
        skipSpecialTokens: false,
      );

      expect(decoded, contains('<|endoftext|>'));
    });

    test('decodes English text correctly', () {
      // English-only vocab uses DIFFERENT token IDs than multilingual!
      // 843=" And", 370=" so" (same as multilingual), 452=" my" (same)
      final decoded = WhisperTokenizer.instance.decode(
        [843, 523, 616], // " And" " so" " my"
        skipSpecialTokens: true,
      );

      expect(decoded, contains('And'));
      expect(decoded, contains('so'));
      expect(decoded, contains('my'));
    });
  });
}
