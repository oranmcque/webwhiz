import {
  WebSocketGateway,
  WebSocketServer,
  OnGatewayConnection,
  OnGatewayDisconnect,
  OnGatewayInit,
  SubscribeMessage,
} from '@nestjs/websockets';
import { Logger, Inject, forwardRef } from '@nestjs/common';
import { Socket } from 'socket.io';
import { Redis } from 'ioredis';
import { ChatQueryAnswer } from './knowledgebase.schema';
import { AppConfigService } from '../common/config/appConfig.service';
import { ChatbotService } from './chatbot/chatbot.service';
import { OfflineMsgService } from './offline-msg/offline-msg.service';

const SESSION_KB_MAPPING = 'sessionKbMapping';

@WebSocketGateway({
  cors: {
    origin: '*',
  },
})
export class WebSocketChatGateway
  implements OnGatewayConnection, OnGatewayDisconnect, OnGatewayInit
{
  @WebSocketServer() server;
  private readonly logger = new Logger(WebSocketChatGateway.name);
  private redisClient: Redis;

  constructor(
    private appConfig: AppConfigService,
    @Inject(forwardRef(() => ChatbotService))
    private chatbotService: ChatbotService,
    private offlineMsgService: OfflineMsgService,
  ) {
    const redisUrl = this.appConfig.get('redisUrl');

    if (redisUrl) {
      this.redisClient = new Redis(redisUrl);
    } else {
      this.redisClient = new Redis({
        host: this.appConfig.get('redisHost'),
        port: this.appConfig.get('redisPort'),
      });
    }
  }

  afterInit() {
    this.logger.log('Websocket gateway initialized.');
  }

  async handleConnection(socket: Socket) {
    // A client has connected
    const query = socket.handshake.query;
    const sessionId: any = query.id;
    const knowledgeBaseId: any = query.knowledgeBaseId;

    // Set session knowledgeBaseId mapping in REDIS for easy fetching
    this.redisClient.hset(SESSION_KB_MAPPING, sessionId, knowledgeBaseId);

    if (query.isAdmin) {
      this.logger.log('Admin joined:', query);
      // set online admins in redis
      this.redisClient.hset(`onlineAdmins_${knowledgeBaseId}`, sessionId, 1);
      // join knowledgeBase room
      socket.join(knowledgeBaseId);
    } else {
      this.logger.log('New client joined:', query);
      // join session chat room
      socket.join(sessionId);

      socket.to(knowledgeBaseId).emit('user_assigned', sessionId);
    }
  }

  async handleDisconnect(socket: Socket) {
    // A client has disconnected
    const query = socket.handshake.query;
    const sessionId: any = query.id;

    if (query.isAdmin) {
      const knowledgeBaseId = query.knowledgeBaseId;
      // remove online admin from redis
      this.redisClient.hdel(`onlineAdmins_${knowledgeBaseId}`, sessionId);
      this.logger.log('Admin disconnected:', query);
    } else {
      this.logger.log('Client disconnected:', query);
    }
  }

  @SubscribeMessage('admin_chat')
  async onAdminChat(client: Socket, msgData: ChatQueryAnswer) {
    this.logger.log('New admin chat message:', msgData);

    client.to(msgData.sessionId).emit('user_chat', msgData);
    const kbId = await this.redisClient.hget(
      SESSION_KB_MAPPING,
      msgData.sessionId,
    );
    if (kbId) {
      this.server.to(kbId).emit('chat_broadcast', msgData);
    }

    this.chatbotService.saveManualChat(msgData.sessionId, msgData);
  }

  @SubscribeMessage('user_chat')
  async onUserChat(client: Socket, msgData: ChatQueryAnswer) {
    this.logger.log('New user chat message:', msgData);

    const kbId = await this.redisClient.hget(
      SESSION_KB_MAPPING,
      msgData.sessionId,
    );

    if (kbId) {
      client.to(kbId).emit('admin_chat', msgData);
    }

    // save the chat in db
    const knowledgeBaseId = await this.chatbotService.saveManualChat(
      msgData.sessionId,
      msgData,
    );

    const onlineAdmins = await this.redisClient.hgetall(
      `onlineAdmins_${knowledgeBaseId}`,
    );

    if (Object.keys(onlineAdmins).length === 0) {
      // send email if the admin is offline
      this.logger.log('No online admins online!!!!');
      this.offlineMsgService.sendEmailForOfflineManualMessage(
        knowledgeBaseId,
        msgData.msg,
      );
    }
  }
}
