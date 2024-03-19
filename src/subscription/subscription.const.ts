import { Subscription } from '../user/user.schema';

export enum SubscriptionType {
  MONTHLY = 'MONTHLY',
  YEARLY = 'YEARLY',
  LIFETIME = 'LIFETIME',
}

export interface SubscriptionPlanInfo {
  name: string;
  type: SubscriptionType;
  maxChatbots: number;
  maxTokens: number;
  maxPages: number;
  maxChunksPerPage: number;
  maxMessages: number;
}

export const subscriptionPlanData: Record<Subscription, SubscriptionPlanInfo> =
  {
    [Subscription.FREE]: {
      name: 'FREE',
      type: SubscriptionType.MONTHLY,
      maxChatbots: 1,
      maxTokens: 20000,
      maxPages: 25,
      maxChunksPerPage: 50,
      maxMessages: 20,
    },
    [Subscription.BASE_MONTHLY]: {
      name: 'Base',
      type: SubscriptionType.MONTHLY,
      maxChatbots: 5,
      maxTokens: 4000000,
      maxPages: 100,
      maxChunksPerPage: 100,
      maxMessages: 4000,
    },
    [Subscription.STANDARD_MONTHLY]: {
      name: 'Standard',
      type: SubscriptionType.MONTHLY,
      maxChatbots: 10,
      maxTokens: 10000000,
      maxPages: 1000,
      maxChunksPerPage: 100,
      maxMessages: 10000,
    },
    [Subscription.PREMIUM_MONTHLY]: {
      name: 'Premium',
      type: SubscriptionType.MONTHLY,
      maxChatbots: 100,
      maxTokens: 25000000,
      maxPages: 2500,
      maxChunksPerPage: 100,
      maxMessages: 25000,
    },
    [Subscription.ENTERPRISE_MONTHLY]: {
      name: 'Enterprise',
      type: SubscriptionType.MONTHLY,
      maxChatbots: 999,
      maxTokens: 95000000,
      maxPages: 10000,
      maxChunksPerPage: 100,
      maxMessages: 100000,
    },
    [Subscription.BASE_YEARLY]: {
      name: 'Base',
      type: SubscriptionType.YEARLY,
      maxChatbots: 5,
      maxTokens: 4000000,
      maxPages: 100,
      maxChunksPerPage: 100,
      maxMessages: 4000,
    },
    [Subscription.STANDARD_YEARLY]: {
      name: 'Standard',
      type: SubscriptionType.YEARLY,
      maxChatbots: 10,
      maxTokens: 10000000,
      maxPages: 1000,
      maxChunksPerPage: 100,
      maxMessages: 10000,
    },
    [Subscription.PREMIUM_YEARLY]: {
      name: 'Premium',
      type: SubscriptionType.YEARLY,
      maxChatbots: 100,
      maxTokens: 25000000,
      maxPages: 2500,
      maxChunksPerPage: 100,
      maxMessages: 25000,
    },
    [Subscription.ENTERPRISE_YEARLY]: {
      name: 'Enterprise',
      type: SubscriptionType.YEARLY,
      maxChatbots: 999,
      maxTokens: 95000000,
      maxPages: 10000,
      maxChunksPerPage: 100,
      maxMessages: 100000,
    },
    [Subscription.DEMO_ACCOUNT]: {
      name: 'DEMO',
      type: SubscriptionType.MONTHLY,
      maxChatbots: 500,
      maxTokens: 995000000,
      maxPages: 200,
      maxChunksPerPage: 100,
      maxMessages: 99900000,
    },
    // APP SUMO PLANS
    [Subscription.APPSUMO_TIER1]: {
      name: 'App Sumo Tier 1',
      type: SubscriptionType.LIFETIME,
      maxChatbots: 20,
      maxTokens: 1000000,
      maxPages: 2000,
      maxChunksPerPage: 100,
      maxMessages: 1000,
    },
    [Subscription.APPSUMO_TIER2]: {
      name: 'App Sumo Tier 2',
      type: SubscriptionType.LIFETIME,
      maxChatbots: 50,
      maxTokens: 2500000,
      maxPages: 5000,
      maxChunksPerPage: 100,
      maxMessages: 2500,
    },
    [Subscription.APPSUMO_TIER3]: {
      name: 'App Sumo Tier 3',
      type: SubscriptionType.LIFETIME,
      maxChatbots: 500,
      maxTokens: 5000000,
      maxPages: 10000,
      maxChunksPerPage: 100,
      maxMessages: 5000,
    },
    [Subscription.SELF_HOSTED]: {
      name: 'Self Hosted',
      type: SubscriptionType.LIFETIME,
      maxChatbots: 500,
      maxTokens: 5000000,
      maxPages: 100000,
      maxChunksPerPage: 500,
      maxMessages: 1000000,
    },
  };
