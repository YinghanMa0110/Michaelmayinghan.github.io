/**
 * api/chat.js (v3 — 加入个人品味话题的 prompt 调整)
 * --------------------------------------------------------------
 * 跟 v2 的区别:
 *   - system prompt 增加"偏好话题特殊规则":
 *     当问到他喜欢的书/音乐/作家/歌手等,允许 AI 自然展开,
 *     不需要每个细节都拘泥于资料原文。但仍禁止编造他没说过的看法。
 *   - 提高检索 top-k 从 4 → 5,品味话题需要更多上下文
 * --------------------------------------------------------------
 */

import fs from 'node:fs';
import path from 'node:path';

let CACHED_KB = null;

function loadKB() {
  if (CACHED_KB) return CACHED_KB;
  const kbPath = path.join(process.cwd(), 'data', 'kb.json');
  CACHED_KB = JSON.parse(fs.readFileSync(kbPath, 'utf8'));
  return CACHED_KB;
}

function tokenize(text) {
  const tokens = [];
  const lower = text.toLowerCase();

  const enMatches = lower.match(/[a-z0-9][a-z0-9\-]*[a-z0-9]/g) || [];
  tokens.push(...enMatches);

  const chineseChars = text.match(/[\u4e00-\u9fff]+/g) || [];
  for (const seq of chineseChars) {
    for (let i = 0; i + 2 <= seq.length; i++) tokens.push(seq.slice(i, i + 2));
    for (let i = 0; i + 3 <= seq.length; i++) tokens.push(seq.slice(i, i + 3));
    for (const ch of seq) tokens.push(ch);
  }

  return tokens;
}

function scoreChunk(queryTokens, chunk, df, N) {
  let score = 0;
  for (const t of queryTokens) {
    const tf = chunk.tf[t];
    if (!tf) continue;
    const idf = Math.log((N + 1) / ((df[t] || 0) + 1)) + 1;
    score += tf * idf;
  }
  return score / Math.sqrt(chunk.tokenCount);
}

function retrieve(query, lang, k = 5) {
  const kb = loadKB();
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) return [];

  const uniqueQueryTokens = [...new Set(queryTokens)];

  const scored = kb.chunks.map(c => {
    let score = scoreChunk(uniqueQueryTokens, c, kb.df, kb.totalChunks);
    if (c.lang === lang) score *= 1.1;
    return { ...c, score };
  });

  scored.sort((a, b) => b.score - a.score);

  if (scored[0].score === 0) {
    const fallback = kb.chunks.filter(c => c.lang === lang && c.id.includes('profile')).slice(0, 1);
    const others = kb.chunks.filter(c => c.lang === lang).slice(0, k - fallback.length);
    return [...fallback, ...others].map(c => ({ ...c, score: 0 }));
  }

  return scored.slice(0, k);
}

function buildSystemPrompt(lang) {
  const ZH = `你是 Yinghan Ma(马英涵)个人主页 yinghanma.com 的智能策展助理。

人设:
- 第三人称介绍 Yinghan,不要假装自己是 Yinghan 本人。
- 语气直白、克制,带一点工程师的精确感和摄影师的诗意。简短,不啰嗦。
- 不要过度热情,不要堆砌赞美形容词。事实优先。

回答规则:

【关于工程项目、研究、教育、技能】
1. 严格基于"参考资料"作答。资料中没有的具体细节(数字、奖项、公司名、邮箱等),直接说"这个我没有可靠资料,建议直接联系 Yinghan:yinghan.ma.mike@gmail.com"。
2. 涉及具体数据(R²、精度、频率、时间常数等)时引用资料原文,不要四舍五入。

【关于个人品味话题——书、音乐、电影、作家、歌手等】
3. 这类话题允许更自然地展开聊。如果资料里提到了某个作家、某首歌,你可以:
   - 介绍 Yinghan 喜欢这位作家/歌手的理由(基于资料)
   - 推荐资料里提到的具体作品
   - 简单聊一两句这本书/这首歌的氛围或主题
   - 邀请访客分享他们的看法(比如"你听过这首吗?"、"你怎么看?")
4. 但仍然禁止:编造 Yinghan 没说过的具体观点(例如不要说"他认为陀思妥耶夫斯基比托尔斯泰更伟大",除非资料明确这么说)。
5. 如果访客提到一个资料里没有的作家/歌手(比如"我喜欢卡夫卡",但资料里只有陀思妥耶夫斯基),不要冷淡地说"没资料"。可以这样回应:"Yinghan 资料里没有提到卡夫卡——但他喜欢陀思妥耶夫斯基,如果你也读这一脉,你们应该聊得来。"自然过桥到资料里有的内容。

【当用户在回答你的问题时】
6. 如果用户的消息看起来是在回答你刚才提出的问题(例如"我是通过…找到这个网站的"),请用温暖、自然的方式回应,像普通对话助理一样,不要去知识库里搜索答案。

【动作指令】
7. 如果用户提到要"看项目"、"看作品"、"看简历"、"联系方式",在回答末尾追加合适的指令标签:
   - 跳转工程项目: [ACTION:GOTO_LAB]
   - 跳转影像作品: [ACTION:GOTO_GALLERY]
   - 跳转简介与 CV: [ACTION:GOTO_ABOUT]
   - 跳转联系方式: [ACTION:GOTO_CONTACT]
   - 切换深色模式: [ACTION:DARK_MODE]
   - 切换浅色模式: [ACTION:LIGHT_MODE]
8. 指令标签严格按 [ACTION:XXX] 格式,放在回答末尾,前面用空格隔开,不要解释。

用中文回答。回复保持在 3-5 句话,不要太长。`;

  const EN = `You are the curator assistant for Yinghan Ma's site yinghanma.com.

Persona:
- Refer to Yinghan in third person. Do NOT pretend to be Yinghan.
- Tone: direct, restrained, with the precision of an engineer and a touch of a photographer's poetry. Short. No fluff.
- No flattery, no piling up adjectives. Facts first.

Rules:

[For engineering projects, research, education, skills]
1. Answer strictly based on "Reference". Specifics not in the references (numbers, awards, companies, emails) — just say "I don't have reliable info on that — best to contact Yinghan: yinghan.ma.mike@gmail.com."
2. When citing data (R², precision, frequency, etc.), quote verbatim. No rounding.

[For personal taste — books, music, films, writers, artists]
3. These topics allow more natural conversation. If the references mention a writer or song, you can:
   - Explain why Yinghan likes them (based on references)
   - Recommend specific works mentioned in references
   - Briefly discuss the mood or theme of a book/song
   - Invite the visitor to share their view ("Have you read this?", "What do you think?")
4. Still forbidden: fabricating opinions Yinghan hasn't expressed (e.g., don't say "he thinks Dostoevsky is greater than Tolstoy" unless the references say so).
5. If a visitor mentions a writer/artist NOT in the references (e.g. "I love Kafka" when references only have Dostoevsky), don't be cold. Try: "The references don't mention Kafka — but Yinghan reads Dostoevsky, so if you're in that lineage, you'd probably get along." Bridge naturally to what IS in the references.

[When the user is answering your question]
6. If the user's message appears to be answering a personal question (e.g. "I found this site through..."), respond warmly and naturally as a conversational assistant, not as a knowledge retrieval system. Do not search the knowledge base for these responses.

[Action tags]
7. If the user wants to "see projects", "see work", "see CV", "contact", append the appropriate tag at the end:
   - Engineering: [ACTION:GOTO_LAB]
   - Gallery: [ACTION:GOTO_GALLERY]
   - About/CV: [ACTION:GOTO_ABOUT]
   - Contact: [ACTION:GOTO_CONTACT]
   - Dark mode: [ACTION:DARK_MODE]
   - Light mode: [ACTION:LIGHT_MODE]
8. Tags must follow [ACTION:XXX] format, at the very end, separated by a space.

Respond in English. Keep replies to 3-5 sentences, not too long.`;

  return lang === 'zh' ? ZH : EN;
}

function formatContext(chunks, lang) {
  const header = lang === 'zh' ? '【参考资料】' : '[Reference]';
  return header + '\n\n' + chunks.map((c, i) =>
    `--- 片段 ${i + 1} (id: ${c.id}, 相关度: ${c.score.toFixed(3)}) ---\n${c.title}\n\n${c.content}`
  ).join('\n\n');
}

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { message, lang = 'zh', history = [] } = req.body || {};
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'message is required' });
    }

    // 检索 top-5(品味话题需要更多上下文)
    const topChunks = retrieve(message, lang, 5);

    const systemPrompt = buildSystemPrompt(lang);
    const contextBlock = formatContext(topChunks, lang);

    const messages = [
      { role: 'system', content: systemPrompt },
      ...history.slice(-4),
      {
        role: 'user',
        content: `${contextBlock}\n\n---\n\n${lang === 'zh' ? '用户问题' : 'User question'}: ${message}`,
      },
    ];

    const chatRes = await fetch('https://api.deepseek.com/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'deepseek-chat',
        messages,
        // 品味话题给点温度,生硬的事实问答用低温度——这里 0.5 折中
        temperature: 0.5,
        max_tokens: 700,
        stream: false,
      }),
    });

    if (!chatRes.ok) {
      const errText = await chatRes.text();
      throw new Error(`Chat API ${chatRes.status}: ${errText}`);
    }

    const chatData = await chatRes.json();
    const reply = chatData.choices[0].message.content;

    return res.status(200).json({
      reply,
      sources: topChunks.map(c => ({
        id: c.id,
        title: c.title,
        lang: c.lang,
        score: Number(c.score.toFixed(3)),
      })),
    });

  } catch (e) {
    console.error('[/api/chat] error:', e);
    return res.status(500).json({
      error: 'internal_error',
      message: e.message,
    });
  }
}
