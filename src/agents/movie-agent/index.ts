import express, { Response } from "express";
import { v4 as uuidv4 } from 'uuid'; // For generating unique IDs
import cors from "cors";

import {
  AgentCard,
  Task,
  TaskState,
  TaskStatusUpdateEvent,
  TextPart,
  Message
} from "@a2a-js/sdk";
import {
  InMemoryTaskStore,
  TaskStore,
  A2AExpressApp,
  AgentExecutor,
  RequestContext,
  ExecutionEventBus,
  DefaultRequestHandler,
} from "@a2a-js/sdk/server";
import { MessageData } from "genkit";
import { ai } from "./genkit.js";
import { searchMovies, searchPeople } from "./tools.js";

if (!process.env.GEMINI_API_KEY || !process.env.TMDB_API_KEY) {
  console.error("GEMINI_API_KEY and TMDB_API_KEY environment variables are required")
  process.exit(1);
}

// Simple store for contexts
const contexts: Map<string, Message[]> = new Map();

// Load the Genkit prompt
const movieAgentPrompt = ai.prompt('movie_agent');

/**
 * MovieAgentExecutor implements the agent's core logic.
 */
class MovieAgentExecutor implements AgentExecutor {
  private cancelledTasks = new Set<string>();

  public cancelTask = async (
    taskId: string,
    eventBus: ExecutionEventBus,
  ): Promise<void> => {
    this.cancelledTasks.add(taskId);
    // The execute loop is responsible for publishing the final state
  };

  async execute(
    requestContext: RequestContext,
    eventBus: ExecutionEventBus
  ): Promise<void> {
    const userMessage = requestContext.userMessage;
    const existingTask = requestContext.task;

    // Determine IDs for the task and context
    const taskId = existingTask?.id || uuidv4();
    const contextId = userMessage.contextId || existingTask?.contextId || uuidv4(); // DefaultRequestHandler should ensure userMessage.contextId

    console.log(
      `[MovieAgentExecutor] Processing message ${userMessage.messageId} for task ${taskId} (context: ${contextId})`
    );

    // 1. Publish initial Task event if it's a new task
    if (!existingTask) {
      const initialTask: Task = {
        kind: 'task',
        id: taskId,
        contextId: contextId,
        status: {
          state: "submitted",
          timestamp: new Date().toISOString(),
        },
        history: [userMessage], // Start history with the current user message
        metadata: userMessage.metadata, // Carry over metadata from message if any
      };
      eventBus.publish(initialTask);
    }

    // 2. Publish "working" status update
    const workingStatusUpdate: TaskStatusUpdateEvent = {
      kind: 'status-update',
      taskId: taskId,
      contextId: contextId,
      status: {
        state: "working",
        message: {
          kind: 'message',
          role: 'agent',
          messageId: uuidv4(),
          parts: [{ kind: 'text', text: 'Processing your question, hang tight!' }],
          taskId: taskId,
          contextId: contextId,
        },
        timestamp: new Date().toISOString(),
      },
      final: false,
    };
    eventBus.publish(workingStatusUpdate);

    // 3. Prepare messages for Genkit prompt
    const historyForGenkit = contexts.get(contextId) || [];
    if (!historyForGenkit.find(m => m.messageId === userMessage.messageId)) {
      historyForGenkit.push(userMessage);
    }
    contexts.set(contextId, historyForGenkit)

    const messages: MessageData[] = historyForGenkit
      .map((m) => ({
        role: (m.role === 'agent' ? 'model' : 'user') as 'user' | 'model',
        content: m.parts
          .filter((p): p is TextPart => p.kind === 'text' && !!(p as TextPart).text)
          .map((p) => ({
            text: (p as TextPart).text,
          })),
      }))
      .filter((m) => m.content.length > 0);

    if (messages.length === 0) {
      console.warn(
        `[MovieAgentExecutor] No valid text messages found in history for task ${taskId}.`
      );
      const failureUpdate: TaskStatusUpdateEvent = {
        kind: 'status-update',
        taskId: taskId,
        contextId: contextId,
        status: {
          state: "failed",
          message: {
            kind: 'message',
            role: 'agent',
            messageId: uuidv4(),
            parts: [{ kind: 'text', text: 'No message found to process.' }],
            taskId: taskId,
            contextId: contextId,
          },
          timestamp: new Date().toISOString(),
        },
        final: true,
      };
      eventBus.publish(failureUpdate);
      return;
    }

    const goal = existingTask?.metadata?.goal as string | undefined || userMessage.metadata?.goal as string | undefined;

    try {
      // 4. Run the Genkit prompt
      const response = await movieAgentPrompt(
        { goal: goal, now: new Date().toISOString() },
        {
          messages,
          tools: [searchMovies, searchPeople],
        }
      );

      // Check if the request has been cancelled
      if (this.cancelledTasks.has(taskId)) {
        console.log(`[MovieAgentExecutor] Request cancelled for task: ${taskId}`);

        const cancelledUpdate: TaskStatusUpdateEvent = {
          kind: 'status-update',
          taskId: taskId,
          contextId: contextId,
          status: {
            state: "canceled",
            timestamp: new Date().toISOString(),
          },
          final: true, // Cancellation is a final state
        };
        eventBus.publish(cancelledUpdate);
        return;
      }

      const responseText = response.text; // Access the text property using .text()
      console.info(`[MovieAgentExecutor] Prompt response: ${responseText}`);
      const lines = responseText.trim().split('\n');
      const finalStateLine = lines.at(-1)?.trim().toUpperCase();
      const agentReplyText = lines.slice(0, lines.length - 1).join('\n').trim();

      let finalA2AState: TaskState = "unknown";

      if (finalStateLine === 'COMPLETED') {
        finalA2AState = "completed";
      } else if (finalStateLine === 'AWAITING_USER_INPUT') {
        finalA2AState = "input-required";
      } else {
        console.warn(
          `[MovieAgentExecutor] Unexpected final state line from prompt: ${finalStateLine}. Defaulting to 'completed'.`
        );
        finalA2AState = "completed"; // Default if LLM deviates
      }

      // 5. Publish final task status update
      const agentMessage: Message = {
        kind: 'message',
        role: 'agent',
        messageId: uuidv4(),
        parts: [{ kind: 'text', text: agentReplyText || "Completed." }], // Ensure some text
        taskId: taskId,
        contextId: contextId,
      };
      historyForGenkit.push(agentMessage);
      contexts.set(contextId, historyForGenkit)

      const finalUpdate: TaskStatusUpdateEvent = {
        kind: 'status-update',
        taskId: taskId,
        contextId: contextId,
        status: {
          state: finalA2AState,
          message: agentMessage,
          timestamp: new Date().toISOString(),
        },
        final: true,
      };
      eventBus.publish(finalUpdate);

      console.log(
        `[MovieAgentExecutor] Task ${taskId} finished with state: ${finalA2AState}`
      );

    } catch (error: any) {
      console.error(
        `[MovieAgentExecutor] Error processing task ${taskId}:`,
        error
      );
      const errorUpdate: TaskStatusUpdateEvent = {
        kind: 'status-update',
        taskId: taskId,
        contextId: contextId,
        status: {
          state: "failed",
          message: {
            kind: 'message',
            role: 'agent',
            messageId: uuidv4(),
            parts: [{ kind: 'text', text: `Agent error: ${error.message}` }],
            taskId: taskId,
            contextId: contextId,
          },
          timestamp: new Date().toISOString(),
        },
        final: true,
      };
      eventBus.publish(errorUpdate);
    }
  }
}

// --- Server Setup ---

const movieAgentCard: AgentCard = {
  name: 'Movie Agent',
  description: 'An agent that can answer questions about movies and actors using TMDB.',
  // Adjust the base URL and port as needed. /a2a is the default base in A2AExpressApp
  url: 'http://localhost:41241/', // Example: if baseUrl in A2AExpressApp 
  provider: {
    organization: 'A2A Samples',
    url: 'https://example.com/a2a-samples' // Added provider URL
  },
  version: '0.0.2', // Incremented version
  capabilities: {
    streaming: true, // The new framework supports streaming
    pushNotifications: false, // Assuming not implemented for this agent yet
    stateTransitionHistory: true, // Agent uses history
  },
  // authentication: null, // Property 'authentication' does not exist on type 'AgentCard'.
  securitySchemes: undefined, // Or define actual security schemes if any
  security: undefined,
  defaultInputModes: ['text'],
  defaultOutputModes: ['text', 'task-status'], // task-status is a common output mode
  skills: [
    {
      id: 'general_movie_chat',
      name: 'General Movie Chat',
      description: 'Answer general questions or chat about movies, actors, directors.',
      tags: ['movies', 'actors', 'directors'],
      examples: [
        'Tell me about the plot of Inception.',
        'Recommend a good sci-fi movie.',
        'Who directed The Matrix?',
        'What other movies has Scarlett Johansson been in?',
        'Find action movies starring Keanu Reeves',
        'Which came out first, Jurassic Park or Terminator 2?',
      ],
      inputModes: ['text'], // Explicitly defining for skill
      outputModes: ['text', 'task-status'] // Explicitly defining for skill
    },
  ],
  supportsAuthenticatedExtendedCard: false,
};

// ---- helpers ----
function readClientPrincipal(req) {
  const b64 = req.headers["x-ms-client-principal"];
  if (!b64) return null;
  const json = Buffer.from(b64, "base64").toString("utf8");
  return JSON.parse(json); // { userId, userDetails, identityProvider, claims: [...] }
}

// Azure Easy Auth can issue different token headers if configured:
// - X-MS-TOKEN-AAD-ACCESS-TOKEN (for AAD resource/scopes)
// - X-MS-TOKEN-AAD-ID-TOKEN
// - X-MS-TOKEN-MICROSOFTACCOUNT-ACCESS-TOKEN (etc., for other providers)
function readTokens(req) {
  const headers = req.headers;
  return {
    aadAccessToken: headers["x-ms-token-aad-access-token"] || null,
    aadIdToken: headers["x-ms-token-aad-id-token"] || null,
  };
}

async function main() {
  // 1. Create TaskStore
  const taskStore: TaskStore = new InMemoryTaskStore();

  // 2. Create AgentExecutor
  const agentExecutor: AgentExecutor = new MovieAgentExecutor();

  // 3. Create DefaultRequestHandler
  const requestHandler = new DefaultRequestHandler(
    movieAgentCard,
    taskStore,
    agentExecutor
  );

  // 4. Create and setup A2AExpressApp
  const appBuilder = new A2AExpressApp(requestHandler);
  const expressApp = express();

  // Enable CORS for all routes and origins before registering routes
  expressApp.use(cors({
    origin: "*",              // allow any origin
    methods: ["GET", "POST"],  // adjust if you need PUT, DELETE, etc.
    allowedHeaders: ["Content-Type", "Authorization"]  // add more if needed
  }));
  expressApp.options("*", cors());

  // Convenience: trigger login (optionally request specific scopes)
  expressApp.get("/login", (req, res) => {
    // Accept ?scopes=... so you can request tokens for Graph or your API
    const scopes = req.query.scopes
      ? `?scopes=${encodeURIComponent(String(req.query.scopes))}`
      : "";
    res.redirect(`/.auth/login/aad${scopes}`);
  });

  expressApp.get("/logout", (_req, res) => res.redirect("/.auth/logout"));

  // Who am I? (decode the Easy Auth header)
  // ignore tslint warning for now
  // @ts-ignore
  expressApp.get("/auth/me", (req, res) => {
    const me = readClientPrincipal(req);
    if (!me) return res.status(401).json({ error: "Not authenticated" });
    res.json(me);
  });

  // Give me the token(s) that Easy Auth placed on the request
  // ignore tslint warning for now
  // @ts-ignore
  expressApp.get("/auth/token", (req, res) => {
    const me = readClientPrincipal(req);
    if (!me) return res.status(401).json({ error: "Not authenticated" });

    const { aadAccessToken, aadIdToken } = readTokens(req);

    if (!aadAccessToken && !aadIdToken) {
      return res.status(400).json({
        error:
          "No tokens found on the request. Ensure Token Store is enabled and that you requested scopes/audiences in the Authentication blade.",
        tips: {
          enableTokenStore: "Portal → Authentication → Token Store = Enabled",
          requestScopes:
            "Login via /login?scopes=<space-separated scopes, url-encoded>",
          exampleScopes:
            "api://<your-api-client-id>/user_impersonation openid profile offline_access",
        },
      });
    }

    res.json({
      user: { id: me.userId, name: me.userDetails, provider: me.identityProvider },
      tokens: {
        access_token: aadAccessToken, // send only what you actually need
        id_token: aadIdToken,
      },
    });
  });

  // Optional: force-refresh session with Easy Auth (if your access token expired)
  expressApp.post("/auth/refresh", (_req, res) => {
    res.redirect(307, "/.auth/refresh"); // Easy Auth refresh endpoint
  });

  // ignore the deprecation warning for now
  // @ts-ignore
  appBuilder.setupRoutes(expressApp);

  // 5. Start the server 41241
  const PORT = Number(process.env.PORT) || 41241;
  expressApp.listen(PORT, "0.0.0.0", () => {
    console.log(`[MovieAgent] Server using new framework started on http://localhost:${PORT}`);
    console.log(`[MovieAgent] Agent Card: http://localhost:${PORT}/.well-known/agent.json`);
    console.log('[MovieAgent] Press Ctrl+C to stop the server');
  });
}

main().catch(console.error);

